import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';

interface Scan {
  id: string;
  domain: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  createdAt: string;
  completedAt?: string;
}

const ScanList: React.FC = () => {
  const [scans, setScans] = useState<Scan[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  
  useEffect(() => {
    // Fetch scans from API
    const fetchScans = async () => {
      try {
        const response = await fetch('/api/scans');
        if (!response.ok) throw new Error('Failed to fetch scans');
        const data = await response.json();
        setScans(data.scans);
      } catch (error) {
        console.error('Error fetching scans:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchScans();
    // Set up polling for scan status updates
    const intervalId = setInterval(fetchScans, 5000);
    return () => clearInterval(intervalId);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'queued': return 'bg-gray-100 text-gray-800';
      case 'running': return 'bg-blue-100 text-blue-800';
      case 'completed': return 'bg-green-100 text-green-800';
      case 'failed': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
        <span className="ml-2">Loading scans...</span>
      </div>
    );
  }

  return (
    <div className="bg-white shadow overflow-hidden sm:rounded-md">
      <ul className="divide-y divide-gray-200">
        {scans.length === 0 ? (
          <li className="px-6 py-4 text-center text-gray-500">
            No scans found. Start a new scan to get started.
          </li>
        ) : (
          scans.map((scan) => (
            <li key={scan.id} className="px-6 py-4 hover:bg-gray-50">
              <div className="flex items-center justify-between cursor-pointer" onClick={() => router.push(`/reports/${scan.id}`)}>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {scan.domain}
                  </p>
                  <p className="text-sm text-gray-500">
                    {new Date(scan.createdAt).toLocaleString()}
                  </p>
                </div>
                <div className="flex-shrink-0">
                  <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getStatusColor(scan.status)}`}
                  >
                    {status.charAt(0).toUpperCase() + status.slice(1)}
                  </span>
                </div>
              </div>
            </li>
          ))
        )}
      </ul>
    </div>
  );
};

export default ScanList;