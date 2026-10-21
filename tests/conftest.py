# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")
