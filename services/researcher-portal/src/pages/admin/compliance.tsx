/**
 * SOC 2 Compliance Evidence Status Page
 * Admin portal for monitoring compliance automation
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Badge,
  Button,
  Progress,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Alert,
  AlertDescription,
  AlertTitle,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui';
import {
  Shield,
  CheckCircle,
  XCircle,
  Clock,
  Download,
  RefreshCw,
  AlertTriangle,
  FileText,
  Database,
  Cloud,
} from 'lucide-react';
import { format } from 'date-fns';

interface EvidenceStatus {
  last_collection: string | null;
  collection_status: string;
  evidence_types_collected: string[];
  failing_controls: number;
  next_collection: string | null;
  s3_bucket: string;
}

interface ComplianceControl {
  control_id: string;
  name: string;
  status: string;
  evidence_count: number;
  last_evaluated: string;
  remediation?: string;
}

interface ComplianceReport {
  report_date: string;
  overall_status: string;
  evidence_collection_status: string;
  last_collection: string | null;
  evidence_types: string[];
  failing_controls: number;
  soc2_readiness: string;
  next_actions: string[];
}

interface EvidenceFile {
  key: string;
  size: number;
  last_modified: string;
  download_url: string;
}

const ComplianceStatusPage: React.FC = () => {
  const [evidenceStatus, setEvidenceStatus] = useState<EvidenceStatus | null>(null);
  const [controls, setControls] = useState<ComplianceControl[]>([]);
  const [report, setReport] = useState<ComplianceReport | null>(null);
  const [evidenceFiles, setEvidenceFiles] = useState<Record<string, EvidenceFile[]>>({});
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedEvidenceType, setSelectedEvidenceType] = useState('daily-evidence');

  const evidenceTypes = [
    { key: 'daily-evidence', label: 'Daily Evidence', icon: Clock },
    { key: 'iam-evidence', label: 'IAM Changes', icon: Shield },
    { key: 'sbom-evidence', label: 'Container SBOMs', icon: Database },
    { key: 'reports', label: 'Compliance Reports', icon: FileText },
  ];

  useEffect(() => {
    fetchComplianceData();
  }, []);

  const fetchComplianceData = async () => {
    try {
      setLoading(true);
      
      // Fetch all compliance data in parallel
      const [statusRes, controlsRes, reportRes] = await Promise.all([
        fetch('/api/compliance/status'),
        fetch('/api/compliance/controls'),
        fetch('/api/compliance/report'),
      ]);

      if (statusRes.ok) {
        setEvidenceStatus(await statusRes.json());
      }
      
      if (controlsRes.ok) {
        setControls(await controlsRes.json());
      }
      
      if (reportRes.ok) {
        setReport(await reportRes.json());
      }

      // Fetch evidence files for each type
      for (const evidenceType of evidenceTypes) {
        try {
          const filesRes = await fetch(`/api/compliance/evidence/${evidenceType.key}`);
          if (filesRes.ok) {
            const filesData = await filesRes.json();
            setEvidenceFiles(prev => ({
              ...prev,
              [evidenceType.key]: filesData.files
            }));
          }
        } catch (error) {
          console.error(`Failed to fetch ${evidenceType.key} files:`, error);
        }
      }
      
    } catch (error) {
      console.error('Failed to fetch compliance data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await fetchComplianceData();
    setRefreshing(false);
  };

  const handleTriggerCollection = async () => {
    try {
      const response = await fetch('/api/compliance/trigger-collection', {
        method: 'POST',
      });
      
      if (response.ok) {
        const result = await response.json();
        // Show success message
        console.log('Evidence collection triggered:', result);
        // Refresh data after a delay
        setTimeout(fetchComplianceData, 2000);
      }
    } catch (error) {
      console.error('Failed to trigger evidence collection:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success':
      case 'compliant':
        return 'bg-green-100 text-green-800';
      case 'error':
      case 'non_compliant':
        return 'bg-red-100 text-red-800';
      case 'needs_review':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getReadinessColor = (readiness: string) => {
    switch (readiness) {
      case 'green':
        return 'text-green-600';
      case 'yellow':
        return 'text-yellow-600';
      case 'red':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading compliance data...</span>
      </div>
    );
  }

  const compliantControls = controls.filter(c => c.status === 'compliant').length;
  const compliancePercentage = controls.length > 0 ? (compliantControls / controls.length) * 100 : 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">SOC 2 Compliance</h1>
          <p className="text-muted-foreground">
            Monitor evidence collection and control compliance status
          </p>
        </div>
        
        <div className="flex space-x-2">
          <Button
            variant="outline"
            onClick={handleRefresh}
            disabled={refreshing}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          
          <Button onClick={handleTriggerCollection}>
            <Cloud className="h-4 w-4 mr-2" />
            Trigger Collection
          </Button>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">SOC 2 Readiness</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${getReadinessColor(report?.soc2_readiness || 'gray')}`}>
              {report?.soc2_readiness?.toUpperCase() || 'UNKNOWN'}
            </div>
            <p className="text-xs text-muted-foreground">
              Overall compliance status
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Evidence Collection</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              <Badge className={getStatusColor(evidenceStatus?.collection_status || 'unknown')}>
                {evidenceStatus?.collection_status || 'Unknown'}
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground">
              {evidenceStatus?.last_collection 
                ? `Last: ${format(new Date(evidenceStatus.last_collection), 'MMM dd, HH:mm')}`
                : 'Never collected'
              }
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Control Compliance</CardTitle>
            <CheckCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {compliantControls}/{controls.length}
            </div>
            <Progress value={compliancePercentage} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-1">
              {compliancePercentage.toFixed(1)}% compliant
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Failing Controls</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">
              {evidenceStatus?.failing_controls || 0}
            </div>
            <p className="text-xs text-muted-foreground">
              Require immediate attention
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Next Actions Alert */}
      {report?.next_actions && report.next_actions.length > 0 && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Action Required</AlertTitle>
          <AlertDescription>
            <ul className="list-disc list-inside mt-2">
              {report.next_actions.map((action, index) => (
                <li key={index}>{action}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {/* Detailed Tabs */}
      <Tabs defaultValue="controls" className="space-y-4">
        <TabsList>
          <TabsTrigger value="controls">Controls Status</TabsTrigger>
          <TabsTrigger value="evidence">Evidence Files</TabsTrigger>
          <TabsTrigger value="reports">Reports</TabsTrigger>
        </TabsList>

        <TabsContent value="controls" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>SOC 2 Controls</CardTitle>
              <CardDescription>
                Current compliance status for all SOC 2 Type II controls
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Control</TableHead>
                    <TableHead>Name</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Evidence</TableHead>
                    <TableHead>Last Evaluated</TableHead>
                    <TableHead>Remediation</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {controls.map((control) => (
                    <TableRow key={control.control_id}>
                      <TableCell className="font-medium">
                        {control.control_id}
                      </TableCell>
                      <TableCell>{control.name}</TableCell>
                      <TableCell>
                        <Badge className={getStatusColor(control.status)}>
                          {control.status}
                        </Badge>
                      </TableCell>
                      <TableCell>{control.evidence_count}</TableCell>
                      <TableCell>
                        {format(new Date(control.last_evaluated), 'MMM dd, yyyy')}
                      </TableCell>
                      <TableCell>
                        <span className="text-sm text-muted-foreground">
                          {control.remediation || 'None required'}
                        </span>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="evidence" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Evidence Files</CardTitle>
              <CardDescription>
                Browse and download compliance evidence by type
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs value={selectedEvidenceType} onValueChange={setSelectedEvidenceType}>
                <TabsList className="grid w-full grid-cols-4">
                  {evidenceTypes.map((type) => {
                    const Icon = type.icon;
                    return (
                      <TabsTrigger key={type.key} value={type.key} className="flex items-center">
                        <Icon className="h-4 w-4 mr-1" />
                        {type.label}
                      </TabsTrigger>
                    );
                  })}
                </TabsList>

                {evidenceTypes.map((type) => (
                  <TabsContent key={type.key} value={type.key}>
                    <div className="space-y-2">
                      {evidenceFiles[type.key]?.map((file) => (
                        <div
                          key={file.key}
                          className="flex items-center justify-between p-3 border rounded-lg"
                        >
                          <div>
                            <div className="font-medium">
                              {file.key.split('/').pop()}
                            </div>
                            <div className="text-sm text-muted-foreground">
                              {(file.size / 1024).toFixed(2)} KB • {format(new Date(file.last_modified), 'MMM dd, yyyy HH:mm')}
                            </div>
                          </div>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => window.open(file.download_url, '_blank')}
                          >
                            <Download className="h-4 w-4 mr-1" />
                            Download
                          </Button>
                        </div>
                      )) || (
                        <div className="text-center py-8 text-muted-foreground">
                          No evidence files found for {type.label}
                        </div>
                      )}
                    </div>
                  </TabsContent>
                ))}
              </Tabs>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="reports" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Compliance Reports</CardTitle>
              <CardDescription>
                Historical compliance reports and trends
              </CardDescription>
            </CardHeader>
            <CardContent>
              {report && (
                <div className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div>
                      <h4 className="font-medium">Report Date</h4>
                      <p className="text-sm text-muted-foreground">{report.report_date}</p>
                    </div>
                    <div>
                      <h4 className="font-medium">Overall Status</h4>
                      <Badge className={getStatusColor(report.overall_status)}>
                        {report.overall_status}
                      </Badge>
                    </div>
                    <div>
                      <h4 className="font-medium">Evidence Types Collected</h4>
                      <p className="text-sm text-muted-foreground">
                        {report.evidence_types.join(', ') || 'None'}
                      </p>
                    </div>
                    <div>
                      <h4 className="font-medium">Failing Controls</h4>
                      <p className="text-sm text-muted-foreground">{report.failing_controls}</p>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ComplianceStatusPage;