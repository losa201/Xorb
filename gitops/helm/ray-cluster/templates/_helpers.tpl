{{/*
Expand the name of the chart.
*/}}
{{- define "ray-cluster.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "ray-cluster.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "ray-cluster.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "ray-cluster.labels" -}}
helm.sh/chart: {{ include "ray-cluster.chart" . }}
{{ include "ray-cluster.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "ray-cluster.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ray-cluster.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "ray-cluster.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "ray-cluster.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the proper image name
*/}}
{{- define "ray-cluster.image" -}}
{{- $registryName := .imageRoot.registry -}}
{{- $repositoryName := .imageRoot.repository -}}
{{- $tag := .imageRoot.tag | toString -}}
{{- if .global }}
    {{- if .global.imageRegistry }}
        {{- $registryName = .global.imageRegistry -}}
    {{- end -}}
{{- end -}}
{{- if $registryName }}
{{- printf "%s/%s:%s" $registryName $repositoryName $tag -}}
{{- else -}}
{{- printf "%s:%s" $repositoryName $tag -}}
{{- end -}}
{{- end }}

{{/*
Return the proper Docker Image Registry Secret Names
*/}}
{{- define "ray-cluster.imagePullSecrets" -}}
{{- include "common.images.pullSecrets" (dict "images" (list .Values.rayHead.image .Values.rayWorkers.image) "global" .Values.global) -}}
{{- end }}

{{/*
Create the name of the configmap
*/}}
{{- define "ray-cluster.configmapName" -}}
{{- printf "%s-config" (include "ray-cluster.fullname" .) -}}
{{- end }}

{{/*
Create the name of the head service
*/}}
{{- define "ray-cluster.headServiceName" -}}
{{- printf "%s-head-svc" (include "ray-cluster.fullname" .) -}}
{{- end }}

{{/*
Create the name of the worker service
*/}}
{{- define "ray-cluster.workerServiceName" -}}
{{- printf "%s-worker-svc" (include "ray-cluster.fullname" .) -}}
{{- end }}

{{/*
Validate values
*/}}
{{- define "ray-cluster.validateValues" -}}
{{- $messages := list -}}
{{/* Validate Ray head configuration */}}
{{- if not .Values.rayHead.enabled -}}
{{- $messages = append $messages "Ray head must be enabled" -}}
{{- end -}}
{{/* Validate resources */}}
{{- if not .Values.rayHead.resources -}}
{{- $messages = append $messages "Ray head resources must be specified" -}}
{{- end -}}
{{- if and .Values.rayWorkers.enabled (not .Values.rayWorkers.resources) -}}
{{- $messages = append $messages "Ray worker resources must be specified when workers are enabled" -}}
{{- end -}}
{{/* Return messages */}}
{{- if $messages -}}
{{- $validationError := join "; " $messages -}}
{{- fail (printf "Ray cluster validation failed: %s" $validationError) -}}
{{- end -}}
{{- end }}