"""Tests for file filtering and security functions in ingest.py."""

from okb.ingest import (
    is_minified,
    matches_pattern,
    scan_content_for_secrets,
)


class TestMatchesPattern:
    """Tests for matches_pattern function."""

    def test_exact_match(self):
        patterns = [".env", ".gitignore"]
        assert matches_pattern(".env", patterns) == ".env"
        assert matches_pattern(".gitignore", patterns) == ".gitignore"

    def test_glob_pattern(self):
        patterns = ["*.pem", "*.key"]
        assert matches_pattern("server.pem", patterns) == "*.pem"
        assert matches_pattern("private.key", patterns) == "*.key"

    def test_case_insensitive(self):
        patterns = ["*.pem"]
        assert matches_pattern("SERVER.PEM", patterns) == "*.pem"
        assert matches_pattern("Certificate.Pem", patterns) == "*.pem"

    def test_no_match_returns_none(self):
        patterns = ["*.pem", "*.key"]
        assert matches_pattern("document.txt", patterns) is None
        assert matches_pattern("code.py", patterns) is None

    def test_empty_patterns_returns_none(self):
        assert matches_pattern("anything.txt", []) is None

    def test_partial_match_not_sufficient(self):
        patterns = [".env"]
        # Should not match files that just contain .env
        assert matches_pattern(".env.example", patterns) is None

    def test_complex_glob_pattern(self):
        patterns = ["*credentials*"]
        assert matches_pattern("aws_credentials.json", patterns) == "*credentials*"
        assert matches_pattern("credentials.yaml", patterns) == "*credentials*"

    def test_lockfile_patterns(self):
        patterns = ["package-lock.json", "yarn.lock", "poetry.lock"]
        assert matches_pattern("package-lock.json", patterns) == "package-lock.json"
        assert matches_pattern("yarn.lock", patterns) == "yarn.lock"


class TestScanContentForSecrets:
    """Tests for scan_content_for_secrets function."""

    def test_detects_private_key(self):
        content = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
-----END RSA PRIVATE KEY-----"""
        result = scan_content_for_secrets(content)
        assert result == "private key"

    def test_detects_ec_private_key(self):
        content = "-----BEGIN EC PRIVATE KEY-----\nsomekey\n-----END EC PRIVATE KEY-----"
        result = scan_content_for_secrets(content)
        assert result == "private key"

    def test_detects_aws_access_key(self):
        content = "aws_access_key_id = AKIAIOSFODNN7EXAMPLE"
        result = scan_content_for_secrets(content)
        assert result == "AWS access key"

    def test_detects_github_pat(self):
        content = "GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        result = scan_content_for_secrets(content)
        assert result == "GitHub personal access token"

    def test_detects_github_oauth(self):
        content = "token: gho_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        result = scan_content_for_secrets(content)
        assert result == "GitHub OAuth token"

    def test_detects_openai_key(self):
        content = "OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        result = scan_content_for_secrets(content)
        assert result == "OpenAI API key"

    def test_detects_anthropic_key(self):
        content = (
            "ANTHROPIC_API_KEY=sk-ant-api" + "x" * 80  # Anthropic keys are long
        )
        result = scan_content_for_secrets(content)
        assert result == "Anthropic API key"

    def test_clean_content_returns_none(self):
        content = """
# My Project

This is a normal markdown file without any secrets.

```python
def hello():
    print("Hello, World!")
```
"""
        result = scan_content_for_secrets(content)
        assert result is None

    def test_only_scans_first_10kb(self):
        # Secret after 10KB should not be detected
        padding = "x" * 15000
        content = padding + "-----BEGIN RSA PRIVATE KEY-----"
        result = scan_content_for_secrets(content)
        assert result is None

    def test_secret_in_first_10kb_detected(self):
        # Secret within first 10KB should be detected
        content = "-----BEGIN RSA PRIVATE KEY-----" + "x" * 15000
        result = scan_content_for_secrets(content)
        assert result == "private key"


class TestIsMinified:
    """Tests for is_minified function."""

    def test_detects_minified_js(self):
        # Simulated minified JS with many semicolons
        content = "var a=1;var b=2;var c=3;" * 100  # Long line with many semicolons
        assert is_minified(content) is True

    def test_detects_minified_css(self):
        # Simulated minified CSS with many braces
        content = ".a{color:red}.b{color:blue}.c{color:green}" * 50
        assert is_minified(content) is True

    def test_normal_code_not_minified(self):
        content = """function hello() {
    console.log("Hello");
}

function world() {
    console.log("World");
}
"""
        assert is_minified(content) is False

    def test_empty_content_not_minified(self):
        assert is_minified("") is False

    def test_short_lines_not_minified(self):
        content = "short line\nanother short line\nthird line"
        assert is_minified(content) is False

    def test_long_string_without_punctuation_not_minified(self):
        # Long line but not minified code (e.g., long comment or text)
        content = "a " * 1000  # Long line but few semicolons/braces
        assert is_minified(content) is False

    def test_custom_max_line_length(self):
        content = "var a=1;var b=2;" * 30  # ~500 chars
        assert is_minified(content, max_line_length=400) is True
        assert is_minified(content, max_line_length=1000) is False

    def test_only_checks_first_lines(self):
        # Minified content after first few lines should not trigger
        normal_lines = "normal line\n" * 10
        minified = "var a=1;var b=2;" * 100
        content = normal_lines + minified
        assert is_minified(content) is False


class TestFileFilteringIntegration:
    """Integration tests combining multiple filtering functions."""

    def test_sensitive_filenames(self):
        sensitive_files = [
            ".env",
            ".env.local",
            "id_rsa",
            "id_ed25519",
            "private.pem",
            "secret.key",
            "credentials.json",
            ".netrc",
            ".pgpass",
        ]
        patterns = [
            ".env",
            ".env.*",
            "id_rsa",
            "id_ed25519",
            "*.pem",
            "*.key",
            "*credentials*",
            ".netrc",
            ".pgpass",
        ]

        for filename in sensitive_files:
            result = matches_pattern(filename, patterns)
            assert result is not None, f"{filename} should be blocked"

    def test_lockfiles(self):
        lockfiles = [
            "package-lock.json",
            "yarn.lock",
            "poetry.lock",
            "Cargo.lock",
            "Gemfile.lock",
        ]
        patterns = [
            "package-lock.json",
            "yarn.lock",
            "poetry.lock",
            "Cargo.lock",
            "Gemfile.lock",
        ]

        for filename in lockfiles:
            result = matches_pattern(filename, patterns)
            assert result is not None, f"{filename} should be skipped"

    def test_normal_files_pass(self):
        normal_files = [
            "main.py",
            "README.md",
            "config.yaml",
            "index.html",
            "styles.css",
            "app.js",
        ]
        sensitive_patterns = [".env", "*.pem", "*.key", "*credentials*"]

        for filename in normal_files:
            result = matches_pattern(filename, sensitive_patterns)
            assert result is None, f"{filename} should not be blocked"
