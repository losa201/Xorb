-- XORB Vulnerable Database - Intentionally insecure for red team training
-- Company database with realistic data and intentional vulnerabilities

-- Create users table with weak security
CREATE TABLE IF NOT EXISTS users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL,
    password VARCHAR(255) NOT NULL,  -- Plain text passwords
    email VARCHAR(100),
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Insert test users with weak credentials
INSERT INTO users (username, password, email, role) VALUES
('admin', 'admin', 'admin@company.com', 'admin'),
('root', 'root', 'root@company.com', 'admin'),
('test', 'test', 'test@company.com', 'user'),
('guest', 'guest', 'guest@company.com', 'user'),
('user', 'password', 'user@company.com', 'user'),
('demo', '123456', 'demo@company.com', 'user'),
('manager', 'manager', 'manager@company.com', 'manager'),
('employee', 'employee', 'employee@company.com', 'user');

-- Employee records table
CREATE TABLE IF NOT EXISTS employees (
    id INT PRIMARY KEY AUTO_INCREMENT,
    employee_id VARCHAR(10) UNIQUE,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    department VARCHAR(50),
    position VARCHAR(100),
    salary DECIMAL(10,2),
    hire_date DATE,
    ssn VARCHAR(11),  -- Sensitive data stored unsecured
    phone VARCHAR(15),
    address TEXT,
    manager_id INT
);

-- Insert employee data
INSERT INTO employees (employee_id, first_name, last_name, department, position, salary, hire_date, ssn, phone, address, manager_id) VALUES
('EMP001', 'John', 'Smith', 'IT', 'System Administrator', 75000.00, '2020-01-15', '123-45-6789', '555-0101', '123 Main St, City, State', NULL),
('EMP002', 'Jane', 'Doe', 'HR', 'HR Manager', 80000.00, '2019-05-10', '987-65-4321', '555-0102', '456 Oak Ave, City, State', NULL),
('EMP003', 'Bob', 'Johnson', 'Finance', 'Accountant', 65000.00, '2021-03-20', '456-78-9123', '555-0103', '789 Pine St, City, State', NULL),
('EMP004', 'Alice', 'Wilson', 'IT', 'Developer', 70000.00, '2021-07-01', '321-54-9876', '555-0104', '321 Elm St, City, State', 1),
('EMP005', 'Charlie', 'Brown', 'Sales', 'Sales Rep', 55000.00, '2022-01-10', '654-32-1098', '555-0105', '654 Maple Ave, City, State', NULL);

-- Financial records table
CREATE TABLE IF NOT EXISTS financial_records (
    id INT PRIMARY KEY AUTO_INCREMENT,
    account_number VARCHAR(20),
    account_holder VARCHAR(100),
    balance DECIMAL(15,2),
    account_type VARCHAR(20),
    routing_number VARCHAR(9),
    created_date DATE,
    last_transaction DATE
);

-- Insert financial data
INSERT INTO financial_records (account_number, account_holder, balance, account_type, routing_number, created_date, last_transaction) VALUES
('1234567890123456', 'Corporate Main Account', 2500000.00, 'checking', '123456789', '2020-01-01', '2023-12-01'),
('9876543210987654', 'Payroll Account', 500000.00, 'checking', '123456789', '2020-01-01', '2023-12-01'),
('5555444433332222', 'Emergency Fund', 1000000.00, 'savings', '123456789', '2020-01-01', '2023-11-15'),
('1111222233334444', 'Petty Cash', 25000.00, 'checking', '123456789', '2020-01-01', '2023-12-01');

-- Customer data table
CREATE TABLE IF NOT EXISTS customers (
    id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id VARCHAR(20),
    company_name VARCHAR(100),
    contact_name VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(15),
    address TEXT,
    credit_limit DECIMAL(10,2),
    current_balance DECIMAL(10,2),
    payment_terms VARCHAR(20),
    tax_id VARCHAR(20)
);

-- Insert customer data
INSERT INTO customers (customer_id, company_name, contact_name, email, phone, address, credit_limit, current_balance, payment_terms, tax_id) VALUES
('CUST001', 'TechCorp Inc', 'Mike Davis', 'mike@techcorp.com', '555-1001', '1000 Business Blvd, Metro City', 100000.00, 45000.00, 'NET30', '12-3456789'),
('CUST002', 'Global Solutions LLC', 'Sarah Lee', 'sarah@globalsol.com', '555-1002', '2000 Commerce Dr, Business City', 75000.00, 32000.00, 'NET15', '98-7654321'),
('CUST003', 'Innovation Systems', 'David Kim', 'david@innosys.com', '555-1003', '3000 Innovation Way, Tech Valley', 150000.00, 87000.00, 'NET45', '45-6789123');

-- System logs table (contains sensitive operations)
CREATE TABLE IF NOT EXISTS system_logs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id INT,
    action VARCHAR(100),
    table_affected VARCHAR(50),
    details TEXT,
    ip_address VARCHAR(15),
    user_agent TEXT
);

-- Insert some log entries
INSERT INTO system_logs (user_id, action, table_affected, details, ip_address, user_agent) VALUES
(1, 'LOGIN', 'users', 'Admin user logged in', '192.168.1.100', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'),
(1, 'SELECT', 'financial_records', 'Accessed financial data', '192.168.1.100', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'),
(2, 'LOGIN', 'users', 'User logged in', '192.168.1.101', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'),
(1, 'UPDATE', 'employees', 'Updated salary information', '192.168.1.100', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)');

-- Create vulnerable stored procedures
DELIMITER //

CREATE PROCEDURE GetUserByCredentials(IN input_username VARCHAR(50), IN input_password VARCHAR(255))
BEGIN
    -- Vulnerable to SQL injection
    SET @sql = CONCAT('SELECT * FROM users WHERE username = "', input_username, '" AND password = "', input_password, '"');
    PREPARE stmt FROM @sql;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
END//

CREATE PROCEDURE SearchEmployees(IN search_term VARCHAR(100))
BEGIN
    -- Vulnerable to SQL injection
    SET @sql = CONCAT('SELECT * FROM employees WHERE first_name LIKE "%', search_term, '%" OR last_name LIKE "%', search_term, '%"');
    PREPARE stmt FROM @sql;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
END//

DELIMITER ;

-- Create views that expose sensitive data
CREATE VIEW employee_sensitive_data AS
SELECT
    employee_id,
    CONCAT(first_name, ' ', last_name) as full_name,
    department,
    salary,
    ssn,
    phone,
    address
FROM employees;

CREATE VIEW financial_summary AS
SELECT
    account_number,
    account_holder,
    balance,
    routing_number
FROM financial_records;

-- Grant excessive privileges to common users
GRANT ALL PRIVILEGES ON company_db.* TO 'admin'@'%';
GRANT SELECT, INSERT, UPDATE ON company_db.* TO 'guest'@'%';

-- Create backup user with weak credentials
CREATE USER 'backup'@'%' IDENTIFIED BY 'backup123';
GRANT SELECT ON company_db.* TO 'backup'@'%';

-- Add comments indicating this is a vulnerable system
ALTER TABLE users COMMENT = 'User authentication table - passwords stored in plain text for testing';
ALTER TABLE employees COMMENT = 'Employee records - contains sensitive PII data';
ALTER TABLE financial_records COMMENT = 'Financial data - high value target for attackers';

-- Create some triggers that log sensitive operations
DELIMITER //

CREATE TRIGGER log_financial_access
AFTER SELECT ON financial_records
FOR EACH ROW
BEGIN
    INSERT INTO system_logs (action, table_affected, details, ip_address)
    VALUES ('SELECT', 'financial_records', 'Financial data accessed', CONNECTION_ID());
END//

DELIMITER ;

-- Show database structure for easy enumeration
SHOW TABLES;
DESCRIBE users;
DESCRIBE employees;
DESCRIBE financial_records;
