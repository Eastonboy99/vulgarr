"""Serialize inference; never reuse a failed CUDA context or an invalid result."""
import logging
import math
import re
import threading
import time


class InferenceUnavailable(RuntimeError):
    pass


class InferenceGuard:
    def __init__(self, admission_seconds=30):
        self.lock = threading.Lock()
        self.admission_seconds = admission_seconds
        self.quarantined = False
        self.last_success = None
        self.failures = 0
        self.last_error_kind = None
        self.cuda_error_code = None

    def run(self, factory, path):
        if self.quarantined:
            raise InferenceUnavailable('Inference quarantined; process recovery required')
        if not self.lock.acquire(timeout=self.admission_seconds):
            self.quarantined = True
            self.failures += 1
            self.last_error_kind = 'admission_timeout'
            raise InferenceUnavailable('Inference admission deadline exceeded')
        try:
            if self.quarantined:
                raise InferenceUnavailable('Inference quarantined; process recovery required')
            try:
                result = factory().detect(str(path))
            except Exception as error:
                self.failures += 1
                self.last_error_kind = 'inference'
                text = str(error).lower()
                if any(s in text for s in ('cuda', 'cublas', 'cudnn', 'onnxruntime')):
                    self.quarantined = True
                    self.last_error_kind = 'provider'
                    match = re.search(r'cuda failure\s+(\d{1,5})\b', text)
                    self.cuda_error_code = int(match.group(1)) if match else None
                    logging.getLogger(__name__).error(
                        'Inference quarantined: provider failure, CUDA code %s', self.cuda_error_code)
                # Never expose provider exception text, source URLs or paths.
                raise InferenceUnavailable('Inference failed; no detection result available') from None
            if (not isinstance(result, list) or any(
                    not isinstance(row,dict) or not isinstance(row.get('class'),str) or not row['class']
                    or type(row.get('score')) not in (int,float) or not 0<=row['score']<=1
                    or not math.isfinite(row['score']) for row in result)):
                self.failures += 1
                self.last_error_kind = 'invalid_output'
                self.quarantined = True
                raise InferenceUnavailable('Invalid detector output')
            if self.quarantined:
                # Another admission deadline expired while this call was in
                # native code. Its eventual return must not clear quarantine.
                raise InferenceUnavailable('Inference quarantined during execution')
            self.last_success = time.monotonic()
            return result
        finally:
            self.lock.release()

    def status(self):
        return {
            'readiness': 'quarantined' if self.quarantined else 'ready' if self.last_success is not None else 'uninitialized',
            'inference_ready': self.last_success is not None and not self.quarantined,
            'inference_failures': self.failures,
            'last_inference_error_kind': self.last_error_kind,
            'cuda_error_code': self.cuda_error_code,
        }
