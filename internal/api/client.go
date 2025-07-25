package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/xorb/xorbctl/internal/auth"
	"github.com/xorb/xorbctl/internal/config"
	"github.com/xorb/xorbctl/internal/version"
)

// Client handles API communication
type Client struct {
	httpClient  *http.Client
	baseURL     string
	accessToken string
}

// NewClient creates a new API client
func NewClient() (*Client, error) {
	token, err := auth.GetToken()
	if err != nil {
		return nil, fmt.Errorf("authentication required: %w", err)
	}

	return &Client{
		httpClient: &http.Client{
			Timeout: time.Duration(config.GetTimeout()) * time.Second,
		},
		baseURL:     config.GetAPIEndpoint(),
		accessToken: token.AccessToken,
	}, nil
}

// makeRequest performs an HTTP request
func (c *Client) makeRequest(method, endpoint string, body interface{}) (*http.Response, error) {
	var reqBody io.Reader
	
	if body != nil {
		jsonData, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		reqBody = bytes.NewReader(jsonData)
	}

	req, err := http.NewRequest(method, c.baseURL+endpoint, reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.accessToken)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", version.GetUserAgent())

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		
		var errorResp struct {
			Error   string `json:"error"`
			Message string `json:"message"`
		}
		
		if err := json.NewDecoder(resp.Body).Decode(&errorResp); err == nil {
			return nil, fmt.Errorf("API error (%d): %s - %s", 
				resp.StatusCode, errorResp.Error, errorResp.Message)
		}
		
		return nil, fmt.Errorf("API error: %d %s", resp.StatusCode, resp.Status)
	}

	return resp, nil
}

// decodeResponse decodes JSON response into target struct
func (c *Client) decodeResponse(resp *http.Response, target interface{}) error {
	defer resp.Body.Close()
	
	if err := json.NewDecoder(resp.Body).Decode(target); err != nil {
		return fmt.Errorf("failed to decode response: %w", err)
	}
	
	return nil
}

// Asset API methods
func CreateAsset(asset *Asset) (*Asset, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	resp, err := client.makeRequest("POST", "/api/assets", asset)
	if err != nil {
		return nil, err
	}

	var result Asset
	if err := client.decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

func ListAssets(filters *AssetFilters) ([]*Asset, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	resp, err := client.makeRequest("GET", "/api/assets", filters)
	if err != nil {
		return nil, err
	}

	var result []*Asset
	if err := client.decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return result, nil
}

func GetAsset(assetID string) (*Asset, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	resp, err := client.makeRequest("GET", "/api/assets/"+assetID, nil)
	if err != nil {
		return nil, err
	}

	var result Asset
	if err := client.decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

func UpdateAsset(assetID string, updates *AssetUpdate) (*Asset, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	resp, err := client.makeRequest("PATCH", "/api/assets/"+assetID, updates)
	if err != nil {
		return nil, err
	}

	var result Asset
	if err := client.decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

func DeleteAsset(assetID string) error {
	client, err := NewClient()
	if err != nil {
		return err
	}

	_, err = client.makeRequest("DELETE", "/api/assets/"+assetID, nil)
	return err
}

// Scan API methods
func StartScan(request *ScanRequest) (*Scan, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	resp, err := client.makeRequest("POST", "/api/scans", request)
	if err != nil {
		return nil, err
	}

	var result Scan
	if err := client.decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

func ListScans(filters *ScanFilters) ([]*Scan, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	resp, err := client.makeRequest("GET", "/api/scans", filters)
	if err != nil {
		return nil, err
	}

	var result []*Scan
	if err := client.decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return result, nil
}

func GetScan(scanID string) (*Scan, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	resp, err := client.makeRequest("GET", "/api/scans/"+scanID, nil)
	if err != nil {
		return nil, err
	}

	var result Scan
	if err := client.decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

func StopScan(scanID string) error {
	client, err := NewClient()
	if err != nil {
		return err
	}

	_, err = client.makeRequest("POST", "/api/scans/"+scanID+"/stop", nil)
	return err
}

func GetScanLogs(scanID string, tail int) ([]*ScanLog, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	endpoint := fmt.Sprintf("/api/scans/%s/logs?tail=%d", scanID, tail)
	resp, err := client.makeRequest("GET", endpoint, nil)
	if err != nil {
		return nil, err
	}

	var result []*ScanLog
	if err := client.decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return result, nil
}

func GetScanLogsSince(scanID string, since time.Time) ([]*ScanLog, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	endpoint := fmt.Sprintf("/api/scans/%s/logs?since=%s", scanID, since.Format(time.RFC3339))
	resp, err := client.makeRequest("GET", endpoint, nil)
	if err != nil {
		return nil, err
	}

	var result []*ScanLog
	if err := client.decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return result, nil
}

func GetScanTemplates() ([]*ScanTemplate, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	resp, err := client.makeRequest("GET", "/api/scan-templates", nil)
	if err != nil {
		return nil, err
	}

	var result []*ScanTemplate
	if err := client.decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return result, nil
}

// Bounty API methods
func CreateBounty(request *BountyRequest) (*Bounty, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	resp, err := client.makeRequest("POST", "/api/bounties", request)
	if err != nil {
		return nil, err
	}

	var result Bounty
	if err := client.decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

func ListBounties(filters *BountyFilters) ([]*Bounty, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	resp, err := client.makeRequest("GET", "/api/bounties", filters)
	if err != nil {
		return nil, err
	}

	var result []*Bounty
	if err := client.decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return result, nil
}

func GetBounty(bountyID string) (*Bounty, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	resp, err := client.makeRequest("GET", "/api/bounties/"+bountyID, nil)
	if err != nil {
		return nil, err
	}

	var result Bounty
	if err := client.decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

func UpdateBounty(bountyID string, updates *BountyUpdate) (*Bounty, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	resp, err := client.makeRequest("PATCH", "/api/bounties/"+bountyID, updates)
	if err != nil {
		return nil, err
	}

	var result Bounty
	if err := client.decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

func CloseBounty(bountyID, reason string) error {
	client, err := NewClient()
	if err != nil {
		return err
	}

	body := map[string]string{"reason": reason}
	_, err = client.makeRequest("POST", "/api/bounties/"+bountyID+"/close", body)
	return err
}

func SubmitVulnerability(submission *VulnerabilitySubmission) (*Finding, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	resp, err := client.makeRequest("POST", "/api/bounties/submit", submission)
	if err != nil {
		return nil, err
	}

	var result Finding
	if err := client.decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

func ListPayouts(filters *PayoutFilters) ([]*Payout, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	resp, err := client.makeRequest("GET", "/api/payouts", filters)
	if err != nil {
		return nil, err
	}

	var result []*Payout
	if err := client.decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return result, nil
}

// Findings API methods
func ListFindings(filters *FindingsFilters) ([]*Finding, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	resp, err := client.makeRequest("GET", "/api/findings", filters)
	if err != nil {
		return nil, err
	}

	var result []*Finding
	if err := client.decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return result, nil
}

func GetFinding(findingID string) (*Finding, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	resp, err := client.makeRequest("GET", "/api/findings/"+findingID, nil)
	if err != nil {
		return nil, err
	}

	var result Finding
	if err := client.decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

func ExportFindings(request *FindingsExportRequest) ([]byte, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	resp, err := client.makeRequest("POST", "/api/findings/export", request)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	return data, nil
}

func ExportFindingsWithTemplate(request *FindingsTemplateExportRequest) ([]byte, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	resp, err := client.makeRequest("POST", "/api/findings/export/template", request)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	return data, nil
}

func GetFindingsStats(request *FindingsStatsRequest) (*FindingsStats, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}

	resp, err := client.makeRequest("POST", "/api/findings/stats", request)
	if err != nil {
		return nil, err
	}

	var result FindingsStats
	if err := client.decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return &result, nil
}