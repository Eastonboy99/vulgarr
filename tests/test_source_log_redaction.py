"""Signed media URLs must not become bearer credentials in access logs."""
import logging

import pytest

from app.vision.remote_source import SourceCapabilityLogFilter, install_source_log_redaction


@pytest.mark.parametrize('method', ['GET', 'HEAD'])
def test_source_access_log_redacts_capability_without_changing_original_arguments(method):
    args = ('127.0.0.1:1234', method,
            '/internal/source?path=private.mkv&expires=9999999999&signature=secret', '1.1', 206)
    record = logging.LogRecord('uvicorn.access', logging.INFO, '', 0,
                              '%s - "%s %s HTTP/%s" %d', args, None)
    assert SourceCapabilityLogFilter().filter(record)
    assert 'secret' not in record.getMessage()
    assert 'private.mkv' not in record.getMessage()
    assert '/internal/source?[redacted]' in record.getMessage()
    assert 'signature=secret' in args[2]  # Do not mutate request/auth data.


def test_other_access_logs_are_unchanged_and_installation_is_idempotent():
    args = ('127.0.0.1:1234', 'GET', '/queue?filter=active', '1.1', 200)
    record = logging.LogRecord('uvicorn.access', logging.INFO, '', 0,
                              '%s - "%s %s HTTP/%s" %d', args, None)
    assert SourceCapabilityLogFilter().filter(record)
    assert record.args == args
    install_source_log_redaction()
    install_source_log_redaction()
    assert sum(isinstance(f, SourceCapabilityLogFilter)
               for f in logging.getLogger('uvicorn.access').filters) == 1
