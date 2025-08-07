<?php
// Intentionally vulnerable web application for red team training
?>
<!DOCTYPE html>
<html>
<head>
    <title>SecureCorp Portal</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .login-form { background: #f4f4f4; padding: 20px; border-radius: 5px; }
        .admin-panel { background: #ffe6e6; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .search-box { margin: 20px 0; }
        input[type="text"], input[type="password"] { padding: 8px; width: 200px; }
        button { padding: 10px 15px; background: #007cba; color: white; border: none; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <h1>SecureCorp Internal Portal</h1>
        
        <?php
        // Vulnerable login system
        if (isset($_POST['username']) && isset($_POST['password'])) {
            $username = $_POST['username'];
            $password = $_POST['password'];
            
            // SQL Injection vulnerability
            $query = "SELECT * FROM users WHERE username = '$username' AND password = '$password'";
            echo "<div style='color: #666; font-size: 12px;'>Debug: $query</div>";
            
            // Simulate login check with vulnerability
            if ($username == "admin' OR '1'='1" || $username == "admin'--" || strpos($username, "UNION") !== false) {
                echo "<div class='admin-panel'>";
                echo "<h3>🔓 Admin Access Granted!</h3>";
                echo "<p>Welcome, Administrator!</p>";
                echo "<p>System Status: <strong>COMPROMISED</strong></p>";
                echo "<p>Database: users, products, financial_records</p>";
                echo "<p>Admin Functions: <a href='?cmd=ls'>List Files</a> | <a href='?cmd=whoami'>System Info</a></p>";
                echo "</div>";
            } elseif ($username == "admin" && $password == "admin") {
                echo "<div class='admin-panel'>";
                echo "<h3>🔓 Admin Access (Default Credentials)</h3>";
                echo "<p>Warning: Using default credentials!</p>";
                echo "</div>";
            } else {
                echo "<div style='color: red;'>Invalid credentials</div>";
            }
        }
        
        // Command injection vulnerability
        if (isset($_GET['cmd'])) {
            $cmd = $_GET['cmd'];
            echo "<div class='admin-panel'>";
            echo "<h4>Command Output:</h4>";
            echo "<pre style='background: #000; color: #0f0; padding: 10px;'>";
            
            // Dangerous command execution
            $output = shell_exec($cmd . " 2>&1");
            echo htmlspecialchars($output);
            echo "</pre>";
            echo "</div>";
        }
        
        // File inclusion vulnerability
        if (isset($_GET['page'])) {
            $page = $_GET['page'];
            echo "<div class='admin-panel'>";
            echo "<h4>Page Content:</h4>";
            echo "<div style='border: 1px solid #ccc; padding: 10px;'>";
            
            // Dangerous file inclusion
            include($page);
            echo "</div>";
            echo "</div>";
        }
        
        // XSS vulnerability in search
        if (isset($_GET['search'])) {
            $search = $_GET['search'];
            echo "<div class='search-results'>";
            echo "<h3>Search Results for: " . $search . "</h3>";
            echo "<p>No results found for your query.</p>";
            echo "</div>";
        }
        ?>
        
        <div class="login-form">
            <h3>Employee Login</h3>
            <form method="POST">
                <p>
                    <label>Username:</label><br>
                    <input type="text" name="username" placeholder="Enter username">
                </p>
                <p>
                    <label>Password:</label><br>
                    <input type="password" name="password" placeholder="Enter password">
                </p>
                <p>
                    <button type="submit">Login</button>
                </p>
            </form>
            <p style="font-size: 12px; color: #666;">
                Default credentials: admin/admin<br>
                For testing: Try SQL injection payloads
            </p>
        </div>
        
        <div class="search-box">
            <h3>Employee Search</h3>
            <form method="GET">
                <input type="text" name="search" placeholder="Search employees..." value="<?php echo isset($_GET['search']) ? htmlspecialchars($_GET['search']) : ''; ?>">
                <button type="submit">Search</button>
            </form>
            <p style="font-size: 12px; color: #666;">
                Try: &lt;script&gt;alert('XSS')&lt;/script&gt;
            </p>
        </div>
        
        <div style="margin-top: 40px; font-size: 12px; color: #999;">
            <p><strong>Vulnerabilities in this application:</strong></p>
            <ul>
                <li>SQL Injection in login form</li>
                <li>Command Injection via cmd parameter</li>
                <li>Local File Inclusion via page parameter</li>
                <li>Cross-Site Scripting in search</li>
                <li>Default credentials (admin/admin)</li>
                <li>Information disclosure in error messages</li>
            </ul>
            <p>This is an intentionally vulnerable application for security testing.</p>
        </div>
    </div>
</body>
</html>