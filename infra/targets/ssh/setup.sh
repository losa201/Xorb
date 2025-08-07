#!/bin/bash
# XORB SSH Target Setup - Intentionally vulnerable SSH server

echo "Setting up vulnerable SSH target..."

# Install required packages
apt-get update
apt-get install -y openssh-server sudo vim curl wget netcat

# Create vulnerable users with weak passwords
useradd -m -s /bin/bash admin
echo 'admin:admin' | chpasswd
usermod -aG sudo admin

useradd -m -s /bin/bash test
echo 'test:test' | chpasswd

useradd -m -s /bin/bash guest
echo 'guest:guest' | chpasswd

useradd -m -s /bin/bash backup
echo 'backup:backup123' | chpasswd

useradd -m -s /bin/bash service
echo 'service:service' | chpasswd

# Set weak root password
echo 'root:root' | chpasswd

# Configure SSH with vulnerabilities
mkdir -p /var/run/sshd

# Enable root login
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Enable password authentication
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Allow empty passwords (very insecure)
sed -i 's/#PermitEmptyPasswords no/PermitEmptyPasswords yes/' /etc/ssh/sshd_config

# Enable X11 forwarding
sed -i 's/#X11Forwarding yes/X11Forwarding yes/' /etc/ssh/sshd_config

# Enable agent forwarding
sed -i 's/#AllowAgentForwarding yes/AllowAgentForwarding yes/' /etc/ssh/sshd_config

# Enable TCP forwarding
sed -i 's/#AllowTcpForwarding yes/AllowTcpForwarding yes/' /etc/ssh/sshd_config

# Increase MaxAuthTries (allows brute force)
echo "MaxAuthTries 10" >> /etc/ssh/sshd_config

# Set weak banner
echo "Banner /etc/ssh/banner" >> /etc/ssh/sshd_config
cat > /etc/ssh/banner << 'EOF'
*******************************************
* Welcome to SecureCorp SSH Server       *
* Authorized users only                  *
* Default credentials: admin/admin       *
* For support: admin@securecorp.local    *
*******************************************
EOF

# Create some interesting files for post-exploitation
mkdir -p /home/admin/documents
cat > /home/admin/documents/passwords.txt << 'EOF'
# Company Password List
admin:admin
root:root
database:db_password_123
backup:backup123
service_account:service_pass_456
EOF

cat > /home/admin/documents/network_diagram.txt << 'EOF'
Network Infrastructure:
- Web Server: 172.20.0.30 (Apache/PHP)
- Database: 172.20.0.40 (MySQL)
- File Server: 172.20.0.60 (Samba)
- Admin Workstation: 172.20.0.50 (SSH)
- Security Monitor: 172.20.0.20 (Blue Team)
EOF

cat > /home/admin/documents/sensitive_data.txt << 'EOF'
CONFIDENTIAL - Company Financial Data
Q4 Revenue: $5,250,000
Bank Account: 1234-5678-9012-3456
Routing: 123456789
Emergency Contact: CEO Direct Line 555-0001
EOF

# Create SSH keys in predictable locations
mkdir -p /home/admin/.ssh
ssh-keygen -t rsa -f /home/admin/.ssh/id_rsa -N ""
cp /home/admin/.ssh/id_rsa.pub /home/admin/.ssh/authorized_keys

# Set permissive permissions (security flaw)
chmod 644 /home/admin/.ssh/id_rsa
chmod 666 /home/admin/documents/passwords.txt
chown -R admin:admin /home/admin

# Create cron jobs that could be exploited
echo "*/5 * * * * /home/admin/backup_script.sh" | crontab -u admin -

# Create the backup script with vulnerabilities
cat > /home/admin/backup_script.sh << 'EOF'
#!/bin/bash
# Vulnerable backup script
BACKUP_DIR="/tmp/backups"
mkdir -p $BACKUP_DIR

# Backing up sensitive files
cp /home/admin/documents/*.txt $BACKUP_DIR/
cp /etc/passwd $BACKUP_DIR/
cp /etc/shadow $BACKUP_DIR/ 2>/dev/null || true

# Log backup activity
echo "$(date): Backup completed" >> /var/log/backup.log
EOF

chmod +x /home/admin/backup_script.sh

# Create sudo configuration with vulnerabilities
echo "admin ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
echo "backup ALL=(root) NOPASSWD:/bin/cp, /bin/mv" >> /etc/sudoers

# Install backdoor services
cat > /etc/systemd/system/backup-service.service << 'EOF'
[Unit]
Description=Backup Service
After=network.target

[Service]
Type=simple
User=root
ExecStart=/bin/nc -l -p 9999 -e /bin/bash
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable the backdoor service
systemctl enable backup-service
systemctl start backup-service

# Create world-writable directories
mkdir -p /tmp/shared
chmod 777 /tmp/shared

# Add environment variables with sensitive info
echo 'export DB_PASSWORD="db_password_123"' >> /home/admin/.bashrc
echo 'export API_KEY="sk-1234567890abcdef"' >> /home/admin/.bashrc
echo 'export ADMIN_TOKEN="admin_token_xyz789"' >> /home/admin/.bashrc

# Create history with sensitive commands
cat > /home/admin/.bash_history << 'EOF'
mysql -u admin -pdb_password_123
ssh root@172.20.0.40
scp sensitive_file.txt backup@fileserver:/backups/
sudo cat /etc/shadow
vim /home/admin/documents/passwords.txt
netstat -tulpn
ps aux | grep ssh
cat /proc/version
uname -a
find / -perm -4000 2>/dev/null
EOF

chown admin:admin /home/admin/.bash_history
chmod 600 /home/admin/.bash_history

# Start SSH daemon
echo "Starting SSH daemon..."
/usr/sbin/sshd -D