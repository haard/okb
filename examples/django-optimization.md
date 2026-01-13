---
tags: [django, postgresql, performance]
project: student-app
category: backend
status: verified
---

# Django Query Optimization

Notes on optimizing database queries in our student application.

## N+1 Query Problems

The most common performance issue. When iterating over a queryset and accessing related objects:

```python
# Bad: N+1 queries
for student in Student.objects.all():
    print(student.course.name)  # Each iteration hits the database

# Good: 1 query with JOIN
for student in Student.objects.select_related('course'):
    print(student.course.name)  # Course data already loaded
```

## Prefetch for Many-to-Many

Use `prefetch_related` for reverse foreign keys and many-to-many:

```python
# Prefetch all enrollments for each course
courses = Course.objects.prefetch_related('enrollments__student')
```

## Bulk Operations

For creating many records:

```python
# Bad: N insert queries
for data in student_data:
    Student.objects.create(**data)

# Good: 1 bulk insert
Student.objects.bulk_create([
    Student(**data) for data in student_data
])
```

## Indexing Strategy

Key indexes for our schema:

- `student.email` - unique, for login lookups
- `enrollment.student_id, enrollment.course_id` - composite for enrollment checks
- `course.start_date` - for date range queries

## Monitoring

Use Django Debug Toolbar in development to catch query issues early.
In production, log slow queries via PostgreSQL:

```sql
ALTER SYSTEM SET log_min_duration_statement = 100;  -- Log queries > 100ms
```
