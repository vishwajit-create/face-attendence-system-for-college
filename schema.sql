-- ============================================================
-- COLLEGE ATTENDANCE SYSTEM - EXTENDED SCHEMA
-- Multi-class, Multi-teacher, Role-based access
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto; -- For password hashing

-- ============================================================
-- 1. ADMIN USERS (Super Admin + Teachers)
-- ============================================================
CREATE TABLE IF NOT EXISTS admin_users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    email TEXT UNIQUE,
    password_hash TEXT NOT NULL,          -- bcrypt hash
    role TEXT NOT NULL DEFAULT 'teacher', -- 'super_admin' | 'teacher'
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT now(),
    last_login TIMESTAMP
);

-- ============================================================
-- 2. DEPARTMENTS
-- ============================================================
CREATE TABLE IF NOT EXISTS departments (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,            -- e.g. 'Computer Science'
    code TEXT UNIQUE,                     -- e.g. 'CS'
    created_at TIMESTAMP DEFAULT now()
);

-- ============================================================
-- 3. CLASSES (Subjects)
-- ============================================================
CREATE TABLE IF NOT EXISTS classes (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,                   -- e.g. 'Data Structures'
    code TEXT UNIQUE NOT NULL,            -- e.g. 'CS301'
    department_id INT REFERENCES departments(id) ON DELETE SET NULL,
    semester INT,                         -- 1 to 8
    year INT DEFAULT EXTRACT(YEAR FROM now())::INT,
    total_working_days INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT now()
);

-- ============================================================
-- 4. CLASS-TEACHER ASSIGNMENT (Many-to-Many)
-- ============================================================
CREATE TABLE IF NOT EXISTS class_teachers (
    id SERIAL PRIMARY KEY,
    class_id INT NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    teacher_id INT NOT NULL REFERENCES admin_users(id) ON DELETE CASCADE,
    assigned_at TIMESTAMP DEFAULT now(),
    UNIQUE (class_id, teacher_id)
);

-- ============================================================
-- 5. STUDENTS (People who can be enrolled in classes)
-- ============================================================
CREATE TABLE IF NOT EXISTS people (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    roll_no TEXT UNIQUE,
    email TEXT,
    phone TEXT,
    department_id INT REFERENCES departments(id) ON DELETE SET NULL,
    semester INT,
    role TEXT DEFAULT 'student',          -- 'student' | 'faculty' | 'employee'
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT now()
);

-- ============================================================
-- 6. CLASS-STUDENT ENROLLMENT (Which students are in which class)
-- ============================================================
CREATE TABLE IF NOT EXISTS class_students (
    id SERIAL PRIMARY KEY,
    class_id INT NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    person_id INT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    enrolled_at TIMESTAMP DEFAULT now(),
    UNIQUE (class_id, person_id)
);

-- ============================================================
-- 7. FACE EMBEDDINGS
-- ============================================================
CREATE TABLE IF NOT EXISTS enrolled_faces (
    id SERIAL PRIMARY KEY,
    person_id INT REFERENCES people(id) ON DELETE CASCADE,
    embedding VECTOR(512),                -- ArcFace 512-dim L2-normalized
    source_image_path TEXT,
    created_at TIMESTAMP DEFAULT now()
);

-- HNSW index for faster cosine search at scale:
-- CREATE INDEX IF NOT EXISTS enrolled_faces_hnsw_idx ON enrolled_faces USING hnsw (embedding vector_cosine_ops);

-- ============================================================
-- 8. SESSIONS (Each class attendance session)
-- ============================================================
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,                  -- e.g. 'CS301-2026-09-28-10AM'
    class_id INT REFERENCES classes(id) ON DELETE SET NULL,
    teacher_id INT REFERENCES admin_users(id) ON DELETE SET NULL,
    label TEXT,
    started_at TIMESTAMP DEFAULT now(),
    ended_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- ============================================================
-- 9. ATTENDANCE LOG (Final per-student per-session record)
-- ============================================================
CREATE TABLE IF NOT EXISTS attendance_log (
    id SERIAL PRIMARY KEY,
    person_id INT REFERENCES people(id),
    session_id TEXT REFERENCES sessions(id),
    class_id INT REFERENCES classes(id),
    marked_at TIMESTAMP DEFAULT now(),
    confidence FLOAT,
    marked_by TEXT DEFAULT 'face_recognition',  -- 'face_recognition' | 'manual'
    UNIQUE (person_id, session_id)              -- DB-level duplicate prevention
);

-- ============================================================
-- 10. UNKNOWN FACE LOG
-- ============================================================
CREATE TABLE IF NOT EXISTS unknown_face_log (
    id SERIAL PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    class_id INT REFERENCES classes(id),
    captured_at TIMESTAMP DEFAULT now(),
    image_path TEXT
);

-- ============================================================
-- DEFAULT SUPER ADMIN
-- Username: superadmin | Password: Admin@123
-- CHANGE IMMEDIATELY AFTER FIRST LOGIN!
-- ============================================================
INSERT INTO admin_users (username, full_name, email, password_hash, role)
VALUES (
    'superadmin',
    'Super Administrator',
    'admin@college.edu',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMtGGzaDBJ3hV4pq5KXZXIW4Y2',  -- Admin@123
    'super_admin'
) ON CONFLICT DO NOTHING;
