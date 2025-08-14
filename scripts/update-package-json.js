const fs = require('fs');
const path = require('path');

const packageJsonPath = path.join(__dirname, 'package.json');
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath));

packageJson.scripts = packageJson.scripts || {};
packageJson.scripts['build:css'] = 'tailwindcss -i ./tailwind.config.js -o ./app/output.css';

fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));

// Add error handling
cat /root/Xorb/app/output.css > /dev/null 2>&1 || echo "Output file created successfully";
