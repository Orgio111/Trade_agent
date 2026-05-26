{{- /* vim: set filetype=mustache: */ -}}

{{- define "trade-agent.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "trade-agent.fullname" -}}
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

{{- define "trade-agent.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "trade-agent.labels" -}}
helm.sh/chart: {{ include "trade-agent.chart" . }}
{{ include "trade-agent.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "trade-agent.selectorLabels" -}}
app.kubernetes.io/name: {{ include "trade-agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "trade-agent.botSelectorLabels" -}}
app.kubernetes.io/name: {{ include "trade-agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
component: bot
{{- end }}

{{- define "trade-agent.dashboardSelectorLabels" -}}
app.kubernetes.io/name: {{ include "trade-agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
component: dashboard
{{- end }}

{{- define "trade-agent.redisSelectorLabels" -}}
app.kubernetes.io/name: {{ include "trade-agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
component: redis
{{- end }}

{{- define "trade-agent.postgresSelectorLabels" -}}
app.kubernetes.io/name: {{ include "trade-agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
component: postgres
{{- end }}

{{- define "trade-agent.image" -}}
{{- $registry := .Values.global.imageRegistry | default "" -}}
{{- $repo := .Values.image.repository -}}
{{- $tag := .Values.image.tag -}}
{{- if $registry -}}
{{- printf "%s/%s:%s" $registry $repo $tag -}}
{{- else -}}
{{- printf "%s:%s" $repo $tag -}}
{{- end -}}
{{- end -}}
