"""Tests for document ingestion functions in ingest.py."""

from datetime import UTC, date, datetime

from okb.ingest import (
    extract_document_date,
    extract_frontmatter,
    extract_org_metadata,
    extract_org_tags,
    extract_org_todo_items,
    extract_sections_markdown,
    extract_sections_org,
    parse_org_timestamp,
)


class TestExtractDocumentDate:
    """Tests for extract_document_date function."""

    def test_extracts_date_field(self):
        assert extract_document_date({"date": "2024-01-15"}) == "2024-01-15"

    def test_extracts_created_field(self):
        assert extract_document_date({"created": "2024-01-15"}) == "2024-01-15"

    def test_extracts_modified_field(self):
        assert extract_document_date({"modified": "2024-01-15"}) == "2024-01-15"

    def test_extracts_updated_field(self):
        assert extract_document_date({"updated": "2024-01-15"}) == "2024-01-15"

    def test_extracts_pubdate_field(self):
        assert extract_document_date({"pubdate": "2024-01-15"}) == "2024-01-15"

    def test_prefers_date_over_other_fields(self):
        metadata = {
            "date": "2024-01-15",
            "created": "2024-01-10",
            "modified": "2024-01-12",
        }
        assert extract_document_date(metadata) == "2024-01-15"

    def test_handles_datetime_objects(self):
        dt = datetime(2024, 1, 15, 12, 0, 0)
        result = extract_document_date({"date": dt})
        assert result == "2024-01-15T12:00:00"

    def test_returns_none_for_empty_metadata(self):
        assert extract_document_date({}) is None

    def test_returns_none_for_no_date_fields(self):
        assert extract_document_date({"title": "Test", "author": "Me"}) is None


class TestExtractFrontmatter:
    """Tests for extract_frontmatter function."""

    def test_extracts_simple_frontmatter(self):
        content = """---
title: Test
date: 2024-01-15
---

Body content here.
"""
        fm, remaining = extract_frontmatter(content)
        assert fm["title"] == "Test"
        # YAML parses dates as datetime.date objects
        assert fm["date"] == date(2024, 1, 15)
        assert "Body content here." in remaining

    def test_extracts_tags_as_list(self):
        content = """---
tags:
  - python
  - testing
---

Content.
"""
        fm, _ = extract_frontmatter(content)
        assert fm["tags"] == ["python", "testing"]

    def test_no_frontmatter_returns_empty_dict(self):
        content = "# Just a heading\n\nSome content."
        fm, remaining = extract_frontmatter(content)
        assert fm == {}
        assert remaining == content

    def test_unclosed_frontmatter_returns_empty(self):
        content = """---
title: Test
no closing delimiter
"""
        fm, remaining = extract_frontmatter(content)
        assert fm == {}
        assert remaining == content

    def test_invalid_yaml_returns_empty(self):
        content = """---
invalid: yaml: syntax: here
---

Content.
"""
        fm, remaining = extract_frontmatter(content)
        assert fm == {}
        assert remaining == content

    def test_frontmatter_not_at_start_ignored(self):
        content = """Some intro text
---
title: This should not be parsed
---
More content.
"""
        fm, remaining = extract_frontmatter(content)
        assert fm == {}
        assert remaining == content


class TestExtractSectionsMarkdown:
    """Tests for extract_sections_markdown function."""

    def test_extracts_h1_sections(self):
        content = """# Section One

Content one.

# Section Two

Content two.
"""
        sections = extract_sections_markdown(content)
        headers = [h for h, _ in sections]
        assert "Section One" in headers
        assert "Section Two" in headers

    def test_extracts_nested_sections(self):
        content = """# Main

Intro.

## Subsection

Sub content.

### Deep

Deep content.
"""
        sections = extract_sections_markdown(content)
        headers = [h for h, _ in sections]
        assert "Main" in headers
        assert "Subsection" in headers
        assert "Deep" in headers

    def test_content_before_first_header(self):
        content = """Intro paragraph before any header.

# First Section

Section content.
"""
        sections = extract_sections_markdown(content)
        # First section should have None header for content before first header
        assert any(h is None for h, _ in sections)

    def test_empty_content(self):
        assert extract_sections_markdown("") == []

    def test_no_headers(self):
        content = "Just some text\nwith multiple lines\nbut no headers."
        sections = extract_sections_markdown(content)
        assert len(sections) == 1
        assert sections[0][0] is None


class TestExtractOrgMetadata:
    """Tests for extract_org_metadata function."""

    def test_extracts_title(self):
        content = """#+TITLE: My Document
#+AUTHOR: Test Author

* Heading
"""
        metadata, remaining = extract_org_metadata(content)
        assert metadata["title"] == "My Document"
        assert metadata["author"] == "Test Author"

    def test_case_insensitive_keys(self):
        content = """#+Title: Test
#+AUTHOR: Author
#+Date: 2024-01-15
"""
        metadata, _ = extract_org_metadata(content)
        assert "title" in metadata
        assert "author" in metadata
        assert "date" in metadata

    def test_multiple_values_become_list(self):
        content = """#+TAGS: tag1
#+TAGS: tag2
"""
        metadata, _ = extract_org_metadata(content)
        assert metadata["tags"] == ["tag1", "tag2"]

    def test_remaining_content_excludes_metadata(self):
        content = """#+TITLE: Test

* First Heading

Content here.
"""
        _, remaining = extract_org_metadata(content)
        assert "#+TITLE" not in remaining
        assert "* First Heading" in remaining

    def test_no_metadata(self):
        content = """* Just a heading

Some content.
"""
        metadata, remaining = extract_org_metadata(content)
        assert metadata == {}
        assert "* Just a heading" in remaining


class TestExtractOrgTags:
    """Tests for extract_org_tags function."""

    def test_extracts_single_tag(self):
        header, tags = extract_org_tags("Some heading :tag:")
        assert header == "Some heading"
        assert tags == ["tag"]

    def test_extracts_multiple_tags(self):
        header, tags = extract_org_tags("Task heading :python:dev:urgent:")
        assert header == "Task heading"
        assert tags == ["python", "dev", "urgent"]

    def test_no_tags(self):
        header, tags = extract_org_tags("Just a heading")
        assert header == "Just a heading"
        assert tags == []

    def test_preserves_colons_in_heading(self):
        header, tags = extract_org_tags("Meeting: 10:00 AM :meeting:")
        assert header == "Meeting: 10:00 AM"
        assert tags == ["meeting"]

    def test_tags_need_whitespace_before(self):
        # Tags must be separated from heading by whitespace
        header, tags = extract_org_tags("No space:tag:")
        assert header == "No space:tag:"
        assert tags == []


class TestExtractSectionsOrg:
    """Tests for extract_sections_org function."""

    def test_extracts_simple_sections(self):
        content = """* Section One

Content one.

* Section Two

Content two.
"""
        sections = extract_sections_org(content)
        headers = [h for h, _ in sections]
        assert "Section One" in headers
        assert "Section Two" in headers

    def test_extracts_nested_sections(self):
        content = """* Top Level

Top content.

** Subsection

Sub content.

*** Deep

Deep content.
"""
        sections = extract_sections_org(content)
        headers = [h for h, _ in sections]
        assert "Top Level" in headers
        assert "Subsection" in headers
        assert "Deep" in headers

    def test_removes_todo_keywords(self):
        content = """* TODO Task One

Task content.

* DONE Task Two

Done content.
"""
        sections = extract_sections_org(content)
        headers = [h for h, _ in sections]
        assert "Task One" in headers
        assert "Task Two" in headers
        assert "TODO Task One" not in headers

    def test_removes_tags_from_headers(self):
        content = """* Heading :tag1:tag2:

Content.
"""
        sections = extract_sections_org(content)
        header = sections[0][0]
        assert ":tag1:" not in header
        assert header == "Heading"

    def test_skips_property_drawers(self):
        content = """* Heading
:PROPERTIES:
:ID: some-id
:END:

Actual content.
"""
        sections = extract_sections_org(content)
        content_text = sections[0][1]
        assert ":PROPERTIES:" not in content_text
        assert "Actual content" in content_text


class TestParseOrgTimestamp:
    """Tests for parse_org_timestamp function."""

    def test_date_with_day_name(self):
        result = parse_org_timestamp("<2024-01-15 Mon>")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_date_with_time(self):
        result = parse_org_timestamp("<2024-01-15 Mon 10:30>")
        assert result is not None
        assert result.hour == 10
        assert result.minute == 30

    def test_date_only(self):
        result = parse_org_timestamp("<2024-01-15>")
        assert result is not None
        assert result.year == 2024

    def test_inactive_timestamp(self):
        # Square brackets for inactive timestamps
        result = parse_org_timestamp("[2024-01-15 Mon]")
        assert result is not None
        assert result.year == 2024

    def test_has_utc_timezone(self):
        result = parse_org_timestamp("<2024-01-15 Mon>")
        assert result is not None
        assert result.tzinfo == UTC

    def test_invalid_timestamp_returns_none(self):
        assert parse_org_timestamp("not a date") is None
        assert parse_org_timestamp("") is None


class TestExtractOrgTodoItems:
    """Tests for extract_org_todo_items function."""

    def test_extracts_todo_item(self):
        content = """* TODO Write tests

Test content.
"""
        items = extract_org_todo_items(content)
        assert len(items) == 1
        assert items[0].keyword == "TODO"
        assert items[0].heading == "Write tests"

    def test_extracts_done_item(self):
        content = """* DONE Completed task
CLOSED: [2024-01-10 Wed]

Task is done.
"""
        items = extract_org_todo_items(content)
        assert len(items) == 1
        assert items[0].keyword == "DONE"
        assert items[0].closed is not None

    def test_extracts_priority(self):
        content = """* TODO [#A] High priority task

Urgent!
"""
        items = extract_org_todo_items(content)
        assert len(items) == 1
        assert items[0].priority == "A"

    def test_extracts_tags(self):
        content = """* TODO Task with tags :python:testing:

Tagged content.
"""
        items = extract_org_todo_items(content)
        assert len(items) == 1
        assert "python" in items[0].tags
        assert "testing" in items[0].tags

    def test_extracts_deadline(self):
        content = """* TODO Task with deadline
DEADLINE: <2024-02-01 Thu>

Has a deadline.
"""
        items = extract_org_todo_items(content)
        assert len(items) == 1
        assert items[0].deadline is not None
        assert items[0].deadline.month == 2
        assert items[0].deadline.day == 1

    def test_extracts_scheduled(self):
        content = """* TODO Scheduled task
SCHEDULED: <2024-01-20 Sat 10:00>

Scheduled for later.
"""
        items = extract_org_todo_items(content)
        assert len(items) == 1
        assert items[0].scheduled is not None
        assert items[0].scheduled.hour == 10

    def test_ignores_non_todo_headings(self):
        content = """* Regular heading

Not a TODO.

* TODO Actual task

This is a task.
"""
        items = extract_org_todo_items(content)
        assert len(items) == 1
        assert items[0].heading == "Actual task"

    def test_extracts_multiple_items(self):
        content = """* TODO First task

First.

* TODO Second task

Second.

* DONE Third task

Third.
"""
        items = extract_org_todo_items(content)
        assert len(items) == 3

    def test_extracts_waiting_keyword(self):
        content = """* WAITING For review

Waiting.
"""
        items = extract_org_todo_items(content)
        assert len(items) == 1
        assert items[0].keyword == "WAITING"

    def test_extracts_someday_keyword(self):
        content = """* SOMEDAY Learn Rust

Maybe later.
"""
        items = extract_org_todo_items(content)
        assert len(items) == 1
        assert items[0].keyword == "SOMEDAY"

    def test_body_content_collected(self):
        content = """* TODO Multi-line task

Line one.
Line two.
Line three.

* TODO Next task
"""
        items = extract_org_todo_items(content)
        assert len(items) == 2
        assert "Line one" in items[0].content
        assert "Line two" in items[0].content
        assert "Line three" in items[0].content


class TestIntegrationWithFixtures:
    """Integration tests using fixture files."""

    def test_markdown_fixture(self, sample_markdown: str):
        """Test full markdown parsing pipeline with fixture."""
        fm, content = extract_frontmatter(sample_markdown)
        assert fm["title"] == "Test Document"
        # YAML parses dates as datetime.date objects
        assert fm["date"] == date(2024, 1, 15)
        assert fm["project"] == "okb"
        assert "python" in fm["tags"]

        sections = extract_sections_markdown(content)
        headers = [h for h, _ in sections]
        assert "Introduction" in headers
        assert "Section One" in headers

    def test_org_fixture(self, sample_org: str):
        """Test full org parsing pipeline with fixture."""
        metadata, content = extract_org_metadata(sample_org)
        assert metadata["title"] == "Test Org Document"
        assert metadata["author"] == "Test Author"

        items = extract_org_todo_items(sample_org)
        keywords = [item.keyword for item in items]
        assert "TODO" in keywords
        assert "DONE" in keywords
        assert "WAITING" in keywords
        assert "SOMEDAY" in keywords

        # Check priority extraction
        high_priority = [i for i in items if i.priority == "A"]
        assert len(high_priority) == 1
        assert "High priority" in high_priority[0].heading
