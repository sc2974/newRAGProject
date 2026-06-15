Set-Location "$PSScriptRoot\..\frontend"

if (-not (Test-Path ".\node_modules")) {
  npm install --cache "$PSScriptRoot\..\.npm-cache"
}

npm run dev
