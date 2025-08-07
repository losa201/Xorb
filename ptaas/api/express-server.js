const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const { v4: uuidv4 } = require('uuid');

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));
app.use(cors());
app.use(helmet());
app.use(morgan('combined'));

// In-memory storage for demo purposes (replace with DB in production)
const scans = {};
const reports = {};

/**
 * POST /scans
 * Create a new scan
 * 
 * Request Body:
 * {
 *   "target": "example.com",
 *   "scan_type": "full", // full, quick, custom
 *   "options": {
 *     "subdomains": true,
 *     "vulnerabilities": true,
 *     "performance": true
 *   }
 * }
 * 
 * Response:
 * 201 Created
 * {
 *   "id": "uuid",
 *   "target": "example.com",
 *   "status": "queued",
 *   "created_at": "timestamp"
 * }
 */
app.post('/scans', (req, res) => {
  try {
    const { target, scan_type = 'quick', options = {} } = req.body;
    
    // Basic validation
    if (!target) {
      return res.status(400).json({ error: 'Target is required' });
    }
    
    // Generate scan ID
    const id = uuidv4();
    
    // Default options based on scan type
    const scanOptions = {
      subdomains: scan_type === 'full' || options.subdomains === true,
      vulnerabilities: scan_type === 'full' || options.vulnerabilities === true,
      performance: scan_type === 'full' || options.performance === true,
      ...options
    };
    
    // Create scan record
    const scan = {
      id,
      target,
      status: 'queued',
      options: scanOptions,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };
    
    scans[id] = scan;
    
    // In a real implementation, we would now:
    // 1. Add scan to queue for processing
    // 2. Trigger scanning engine
    
    // Return 201 Created with Location header
    res.status(201)
      .header('Location', `/scans/${id}`)
      .json(scan);
  } catch (error) {
    console.error('Error creating scan:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

/**
 * GET /scans/{id}
 * Get scan status and details
 * 
 * Response:
 * 200 OK
 * {
 *   "id": "uuid",
 *   "target": "example.com",
 *   "status": "running|completed|failed",
 *   "progress": 75,
 *   "created_at": "timestamp",
 *   "updated_at": "timestamp"
 * }
 */
app.get('/scans/:id', (req, res) => {
  try {
    const { id } = req.params;
    const scan = scans[id];
    
    if (!scan) {
      return res.status(404).json({ error: 'Scan not found' });
    }
    
    // In a real implementation, we would fetch the latest status from the scanning engine
    // For demo, return the stored scan with simulated progress
    const response = {
      ...scan,
      progress: scan.status === 'completed' ? 100 : scan.status === 'failed' ? 0 : 75,
      updated_at: new Date().toISOString()
    };
    
    res.json(response);
  } catch (error) {
    console.error('Error fetching scan:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

/**
 * GET /reports/{id}
 * Get scan report
 * 
 * Response:
 * 200 OK
 * {
 *   "id": "uuid",
 *   "scan_id": "uuid",
 *   "status": "completed|failed",
 *   "summary": {
 *     "vulnerabilities": 5,
 *     "high_risk": 2,
 *     "medium_risk": 1,
 *     "low_risk": 2
 *   },
 *   "findings": [{
 *     "id": "CVE-2021-1234",
 *     "title": "Vulnerability Title",
 *     "risk": "high|medium|low|info",
 *     "description": "Detailed description",
 *     "remediation": "Remediation steps",
 *     "references": ["https://example.com/ref1"]
 *   }],
 *   "created_at": "timestamp"
 * }
 */
app.get('/reports/:id', (req, res) => {
  try {
    const { id } = req.params;
    const report = reports[id];
    
    if (!report) {
      // Check if scan exists but report not generated yet
      const scan = scans[id];
      if (scan) {
        // In a real implementation, we would check if the scan completed
        // and trigger report generation if needed
        return res.status(202).json({ 
          message: 'Report not ready yet',
          scan_id: id,
          status: scan.status
        });
      }
      
      return res.status(404).json({ error: 'Report not found' });
    }
    
    res.json(report);
  } catch (error) {
    console.error('Error fetching report:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

/**
 * POST /integrations/{type}
 * Configure third-party integrations
 * 
 * Request Body:
 * {
 *   "config": {
 *     // Integration-specific configuration
 *   }
 * }
 * 
 * Response:
 * 200 OK
 * {
 *   "integration_type": "slack",
 *   "status": "configured",
 *   "updated_at": "timestamp"
 * }
 */
app.post('/integrations/:type', (req, res) => {
  try {
    const { type } = req.params;
    const { config } = req.body;
    
    // Validate integration type
    const validIntegrations = ['slack', 'jira', 'teams', 'email'];
    if (!validIntegrations.includes(type)) {
      return res.status(400).json({ 
        error: `Invalid integration type. Must be one of: ${validIntegrations.join(',')}` 
      });
    }
    
    // Basic validation
    if (!config) {
      return res.status(400).json({ error: 'Config is required' });
    }
    
    // In a real implementation, we would:
    // 1. Validate integration-specific config
    // 2. Test connection
    // 3. Store encrypted credentials
    
    // For demo, just return success
    res.json({
      integration_type: type,
      status: 'configured',
      updated_at: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error configuring integration:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

/**
 * GET /metrics
 * Get system metrics (for Prometheus scraping)
 * 
 * Response:
 * 200 OK
 * # HELP active_scans Number of active scans
 * # TYPE active_scans gauge
 * active_scans 5
 * # HELP completed_scans Total number of completed scans
 * # TYPE completed_scans counter
 * completed_scans 123
 */
app.get('/metrics', (req, res) => {
  try {
    // In a real implementation, we would collect metrics from various sources
    // For demo, just return some sample metrics
    const activeScans = Object.values(scans).filter(s => s.status === 'running').length;
    const completedScans = Object.values(scans).filter(s => s.status === 'completed').length;
    
    res.setHeader('Content-Type', 'text/plain');
    res.write('# HELP active_scans Number of active scans\n');
    res.write('# TYPE active_scans gauge\n');
    res.write(`active_scans ${activeScans}\n\n`);
    
    res.write('# HELP completed_scans Total number of completed scans\n');
    res.write('# TYPE completed_scans counter\n');
    res.write(`completed_scans ${completedScans}\n\n`);
    
    res.end();
  } catch (error) {
    console.error('Error fetching metrics:', error);
    res.status(500).end('Internal server error');
  }
});

/**
 * Error handling middleware
 */
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: err.message
  });
});

// Start the server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on port ${PORT}`);
});

module.exports = app; // For testing purposes

// To run the server:
// 1. npm init -y
// 2. npm install express cors helmet morgan uuid
// 3. node express-server.js
// 4. Use curl or Postman to test endpoints

// Example curl commands:
// Create scan:
// curl -X POST http://localhost:3000/scans -H "Content-Type: application/json" -d '{"target": "example.com"}'
// Get scan status:
// curl http://localhost:3000/scans/<id>
// Configure Slack integration:
// curl -X POST http://localhost:3000/integrations/slack -H "Content-Type: application/json" -d '{"config": {"webhook_url": "https://hooks.slack.com/services/..."}}'