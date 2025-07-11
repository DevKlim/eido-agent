# EIDO Sentinel

<p align="center">
  <img src="static/images/logo_icon_light.png" alt="EIDO Sentinel Logo Light" width="150"/>
</p>

https://eido-sentinel.streamlit.app/

EIDO Sentinel is an AI-powered platform designed to enhance emergency response by intelligently processing, correlating, and analyzing diverse emergency data streams.

---

## ✨ Features

- **Convert Raw Reports:** Transform raw text from various sources (social media, alerts, dispatches) into a contextualized, standardized incident feed using Large Language Models.
- **Upload EIDO JSONs:** Directly upload existing EIDO-formatted JSON files to populate the incident feed.
- **Live Incident Feed:** View a real-time feed of all processed incidents, with the ability to inspect detailed information for each event.
- **Dashboard & Map View:** Visualize incident data through an interactive dashboard and a map-based interface. Filter incidents by type, location, or status.
- **Export to EIDO:** Easily copy or export processed incidents back into the standardized EIDO JSON format for interoperability.

---

## 🚀 Live Demo & Showcase

### Ingest Raw Text and Generate Standardized Incidents
![GIF of raw text ingestion](static/images/image1.gif)

### Upload and View EIDO JSON Files
![GIF of EIDO JSON upload](static/images/image2.gif)

### Interactive Incident Dashboard and Filtering
![GIF of dashboard view](static/images/image3.gif)

### Geospatial Mapping of All Incidents
![GIF of map view](static/images/image4.gif)

### Detailed Incident Inspection and Export
![GIF of incident details](static/images/image5.gif)

---

## 💡 How It Works: From Raw Text to EIDO JSON

EIDO Sentinel can take unstructured text from multiple sources and convert it into a structured EIDO (Emergency Incident Data Object).

### 1. Raw Text Digest

Here are examples of the kind of unstructured data the system can process:

<details>
<summary><strong>Example 1: Police Alert (Twitter)</strong></summary>

```text
(twitter) @CarlsbadPolice 6:40 PM: ABLE helicopter is making announcements in the Bressi Ranch area near El Fuerte St & Gateway Rd. We're searching for a Hispanic male, 5'6", heavyset, wearing a white shirt & dark pants, armed with a knife. If seen, call 911 immediately.
```
</details>

<details>
<summary><strong>Example 2: Fire Evacuation Notice (Sheriff Dept.)</strong></summary>

```text
San Diego Sheriff @SDSheriff #ClaroFire A brush fire is burning near Corte Claro and Paseo Encino in the City of Carlsbad, near the Carlsbad/San Marcos border. An EVACUATION ORDER is in place for the shaded areas in red shown in the maps below. It means everyone in the impacted area must leave immediately. An EVACUATION WARNING is in place for the shaded areas in yellow shown in the maps below. Be prepared to evacuate. If you feel you are in danger, GO! To see maps of the affected areas, visit: https://protect.genasys.com/location?z=14&latlon=33.091377990534724%2C-117.21915553408178 and https://emergencymap.sandiegocounty.gov/index.html. A Temporary Evacuation Point has been opened at: ▪️Stagecoach Park 3420 Camino De Los Coches, Carlsbad This is a developing situation and the information we provide is current at the time of posting.
```
</details>

<details>
<summary><strong>Example 3: Campus Alert</strong></summary>

```json
{
  "alert_title": "01/17/2025 Police Emergency - Possible Armed Suspect",
  "alert_type": "Other",
  "crime_type": "Unspecified",
  "date": "01/17/2025",
  "description": "

Police Emergency - Possible Armed Suspect
email : Webview : Police Emergency - Possible Armed Suspect


UC SAN DIEGO POLICE DEPARTMENT


January 17, 2025






ALL ACADEMICS, STAFF, AND STUDENTS AT UC SAN DIEGO


Police Emergency - Possible Armed Suspect

Triton Alert Notification: 
Emergency at RIMAC/Liontree Area. Avoid north campus if possible. Otherwise, lock doors and stay inside. UCSD responders on scene. More information to follow as it becomes available. Do not call Police Dispatch unless you have information about this incident.


University of California San Diego, 9500 Gilman Drive, La Jolla, CA, 92093

",
  "details_url": "https://adminrecords.ucsd.edu/Notices/2025/2025-1-17-4.html",
  "is_update": false,
  "location_text": "Unknown",
  "precise_location": "La Jolla",
  "suspect_info": "Suspect
email : Webview : Police Emergency - Possible Armed Suspect


UC SAN DIEGO POLICE DEPARTMENT


January 17, 2025






ALL ACADEMICS, STAFF, AND STUDENTS AT UC SAN DIEGO


Police Emergency - Possible Armed Suspect

Triton Alert Notification: 
Emergency at RIMAC/Liontree Area."
}
```
</details>

### 2. Generated EIDO JSON Output

The system also ingests and processes standard EIDO JSONs. Here are a couple of examples:

<details>
<summary><strong>EIDO JSON Example #1: Border Fire</strong></summary>

```json
{
    "$id": "urn:uuid:1a2b3c4d-5e6f-7890-abcd-ef0123456789",
    "messageKind": "Alert",
    "notesComponent": [
        {
            "noteText": "Fire is 5 acres, slow to moderate spread primarily burning in Mexico.",
            "authorReference": {
                "$ref": "person-aabbccdd-eeff-0011-2233-445566778899"
            },
            "noteDateTimeStamp": "2024-05-15T10:30:00-07:00",
            "componentIdentifier": "note-98765432-10fe-dcba-9876-543210fedcba"
        }
    ],
    "agencyComponent": [
        {
            "$id": "agency-11223344-5566-7788-99aa-bbccddeeff00",
            "agencyName": "CALFIRESANDIEGO",
            "agencyIdentifier": "urn:nena:agency:calfire.sandiego"
        }
    ],
    "personComponent": [
        {
            "$id": "person-aabbccdd-eeff-0011-2233-445566778899",
            "personNameText": "CALFIRESANDIEGO",
            "personIdentifier": "CALFIRESANDIEGO_AlertSystem"
        }
    ],
    "incidentComponent": [
        {
            "locationReference": {
                "$ref": "loc-8a7b6c5d-4e3f-2109-fedc-ba9876543210"
            },
            "componentIdentifier": "inc-0a1b2c3d-4e5f-6789-0abc-def012345678",
            "lastUpdateTimeStamp": "2024-05-15T10:30:00-07:00",
            "updatedByAgencyReference": {
                "$ref": "agency-11223344-5566-7788-99aa-bbccddeeff00"
            },
            "incidentTrackingIdentifier": "Border6Fire",
            "incidentTypeCommonRegistryText": "Vegetation Fire",
            "incidentStatusCommonRegistryText": "Contained"
        }
    ],
    "locationComponent": [
        {
            "$id": "loc-8a7b6c5d-4e3f-2109-fedc-ba9876543210",
            "locationByValue": "<?xml version="1.0" encoding="UTF-8"?>
<location xmlns:gml="http://www.opengis.net/gml" xmlns:ca="urn:ietf:params:xml:ns:pidf:geopriv10:civicAddr">
  <gml:Point srsName="urn:ogc:def:crs:EPSG::4326">
    <gml:pos>32.590 -116.495</gml:pos>
  </gml:Point>
  <ca:civicAddress>
    <ca:country>US</ca:country>
    <ca:A1>CA</ca:A1>
    <ca:A3>Campo</ca:A3>
    <ca:LOC>U.S./Mexico border just west of Forrest Gate Rd</ca:LOC>
    <ca:PC>91906</ca:PC>
  </ca:civicAddress>
  <civicAddressText>U.S./Mexico border just west of Forrest Gate Rd, Campo, CA 91906</civicAddressText>
  <locationNotes>No immediate road impact noted on US side; fire primarily burning in Mexico.</locationNotes>
</location>",
            "componentIdentifier": "loc-8a7b6c5d-4e3f-2109-fedc-ba9876543210",
            "lastUpdateTimeStamp": "2024-05-15T10:30:00-07:00",
            "updatedByAgencyReference": {
                "$ref": "agency-11223344-5566-7788-99aa-bbccddeeff00"
            }
        }
    ],
    "lastUpdateTimeStamp": "2024-05-15T10:30:00-07:00",
    "eidoMessageIdentifier": "urn:uuid:1a2b3c4d-5e6f-7890-abcd-ef0123456789",
    "sendingSystemIdentifier": "urn:nena:agency:calfire.sandiego"
}
```
</details>

<details>
<summary><strong>EIDO JSON Example #2: Monte Fire</strong></summary>

```json
{
    "$id": "urn:uuid:1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d",
    "messageKind": "Alert",
    "notesComponent": [
        {
            "noteText": "A brush fire is burning near the 15000 block of El Monte Road in Lakeside. An EVACUATION ORDER is in place for affected areas. This is a developing situation. Follow @CALFIRESANDIEGO, @SDSheriff, @AlertSanDiegoCo for updates.",
            "authorReference": {
                "$ref": "person-6f7a8b9c-0d1e-2f3a-4b5c-6d7e8f9a0b1c"
            },
            "noteDateTimeStamp": "2024-05-15T14:30:00-07:00",
            "componentIdentifier": "note-5e6f7a8b-9c0d-1e2f-3a4b-5c6d7e8f9a0b"
        }
    ],
    "agencyComponent": [
        {
            "$id": "agency-4d5e6f7a-8b9c-0d1e-2f3a-4b5c6d7e8f9a",
            "agencyName": "UC San Diego Police Department",
            "agencyIdentifier": "urn:nena:agency:ucsdpd"
        }
    ],
    "personComponent": [
        {
            "$id": "person-6f7a8b9c-0d1e-2f3a-4b5c-6d7e8f9a0b1c",
            "personNameText": "UC San Diego Police Department",
            "personIdentifier": "urn:nena:system:ucsdalerts"
        }
    ],
    "incidentComponent": [
        {
            "locationReference": {
                "$ref": "loc-3c4d5e6f-7a8b-9c0d-1e2f-3a4b5c6d7e8f"
            },
            "componentIdentifier": "inc-2b3c4d5e-6f7a-8b9c-0d1e-2f3a4b5c6d7e",
            "lastUpdateTimeStamp": "2024-05-15T14:30:00-07:00",
            "updatedByAgencyReference": {
                "$ref": "agency-4d5e6f7a-8b9c-0d1e-2f3a-4b5c6d7e8f9a"
            },
            "incidentTrackingIdentifier": "MonteFireAlert",
            "incidentTypeCommonRegistryText": "Vegetation Fire",
            "incidentStatusCommonRegistryText": "Active"
        }
    ],
    "locationComponent": [
        {
            "$id": "loc-3c4d5e6f-7a8b-9c0d-1e2f-3a4b5c6d7e8f",
            "locationByValue": "<?xml version="1.0" encoding="UTF-8"?>
<location xmlns:gml="http://www.opengis.net/gml" xmlns:ca="urn:ietf:params:xml:ns:pidf:geopriv10:civicAddr">
  <gml:Point srsName="urn:ogc:def:crs:EPSG::4326">
    <gml:pos>32.8805 -116.8906</gml:pos>
  </gml:Point>
  <ca:civicAddress>
    <ca:country>US</ca:country>
    <ca:A1>CA</ca:A1>
    <ca:A3>Lakeside</ca:A3>
    <ca:LOC>15000 El Monte Road</ca:LOC>
    <ca:PC>92040</ca:PC>
  </ca:civicAddress>
  <civicAddressText>15000 block of El Monte Road in Lakeside, CA 92040</civicAddressText>
  <locationNotes>An EVACUATION ORDER is in place for the shaded areas in red shown in the maps. It means everyone in the impacted area must leave immediately. To see maps of the affected areas, visit: https://protect.genasys.com/zones/US-CA-XSD-SDC-1467 and http://emergencymap.sandiegocounty.gov/index.html.</locationNotes>
</location>",
            "componentIdentifier": "loc-3c4d5e6f-7a8b-9c0d-1e2f-3a4b5c6d7e8f",
            "lastUpdateTimeStamp": "2024-05-15T14:30:00-07:00",
            "updatedByAgencyReference": {
                "$ref": "agency-4d5e6f7a-8b9c-0d1e-2f3a-4b5c6d7e8f9a"
            }
        }
    ],
    "lastUpdateTimeStamp": "2024-05-15T14:30:00-07:00",
    "eidoMessageIdentifier": "urn:uuid:1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d",
    "sendingSystemIdentifier": "urn:nena:agency:ucsdpd"
}
```
</details>

---

## Running Locally with Docker (Recommended)

This project is fully containerized, allowing for an easy, one-command local setup.

### Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)

### Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/LXString/eido-sentinel.git
    cd eido-sentinel
    ```

2.  **Configure Your Environment:**
    Copy the example environment file. The Docker setup will automatically load variables from `.env`.

    ```bash
    cp .env.example .env
    ```

    **IMPORTANT:** You must edit the new `.env` file to provide your API keys like `GOOGLE_API_KEY` and a unique `GEOCODING_USER_AGENT` with your email. For local use, the default URLs are fine.

3.  **Build and Run with Docker Compose:**
    This command builds the Docker image and starts both the backend API and the Streamlit UI services.

    ```bash
    docker-compose up --build
    ```

    - The `--build` flag is only needed the first time or after code changes.
    - To run in the background (detached mode), add the `-d` flag: `docker-compose up -d --build`.

4.  **Access the Application:**

    - **Streamlit UI:** `http://localhost:8501`
    - **FastAPI Backend Docs:** `http://localhost:8000/docs`

5.  **Stopping the Application:**
    - If running in the foreground, press `Ctrl+C`.
    - If running in the background (`-d`), use: `docker-compose down`

---

## Deployment to Fly.io (Recommended Cloud Host)

This project is configured for a multi-process deployment on [Fly.io](https://fly.io). The `api` service will run on the standard web ports (80/443), and the `ui` service will run on port `8080`.

### Prerequisites

- [Docker](https://www.docker.com/get-started) installed locally.
- A [Fly.io](https://fly.io/docs/hands-on/sign-up/) account.
- `flyctl` command-line tool installed. ([Installation Guide](https://fly.io/docs/hands-on/install-flyctl/))

### Step 1: Initial Launch and Setup

1.  **Login to Fly:**
    ```bash
    flyctl auth login
    ```

---

## Deployment to a Cloud VM such as Oracle Cloud Free Tier or Google Cloud Platform

This guide explains how to deploy the entire application (Backend API and Streamlit UI) to a single cloud Virtual Machine using Docker.

### Step 1: Set Up Your Cloud VM

1.  **Create a VM:**

    - Sign up for a cloud provider like [Oracle Cloud Free Tier](https://www.oracle.com/cloud/free/), AWS, GCP, or DigitalOcean.
    - Create a new Compute Instance (VM). For Oracle, an Ampere A1 (ARM64) instance is a great free option. For OS, choose **Ubuntu 22.04** or later.
    - Make sure you can connect to your VM via SSH using the key you provided during setup.

2.  **Install Docker and Docker Compose:**
    - Connect to your VM via SSH.
    - Follow the official Docker documentation to install Docker Engine and Docker Compose for your OS.
      - [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
      - [Install Docker Compose](https://docs.docker.com/compose/install/)

### Step 2: Configure the Firewall

You must allow incoming traffic on ports `8000` (for the API) and `8501` (for the UI).

- **Oracle Cloud (VCN Security List):**

  1.  In your OCI console, navigate to your Virtual Cloud Network (VCN).
  2.  Go to "Security Lists" and select the one associated with your VM's subnet.
  3.  Click "Add Ingress Rules".
  4.  Create a rule:
      - **Source CIDR:** `0.0.0.0/0` (allows traffic from any IP)
      - **IP Protocol:** TCP
      - **Destination Port Range:** `8000,8501`
  5.  Add the rule.

- **On the VM's Firewall (if active):**
  If `ufw` is active on Ubuntu, run:
  ```bash
  sudo ufw allow 8000/tcp
  sudo ufw allow 8501/tcp
  sudo ufw reload
  ```

### Step 3: Set Up the Application on the VM

1.  **Clone Your Repository:**
    On the VM, clone your project.

    ```bash
    git clone https://github.com/LXString/eido-sentinel.git
    cd eido-sentinel
    ```

2.  **Create the Production `.env` File:**
    Copy the example and then edit it for production.

    ```bash
    cp .env.example .env
    nano .env
    ```

3.  **Edit the `.env` File:**
    This is the most critical step. Set the following variables, replacing `<YOUR_SERVER_PUBLIC_IP>` with your VM's public IP address.

    ```env
    # The public URL of your backend API server.
    API_BASE_URL="http://<YOUR_SERVER_PUBLIC_IP>:8000"

    # The public URL of your Streamlit UI. This is ESSENTIAL for CORS.
    STREAMLIT_APP_URL="http://<YOUR_SERVER_PUBLIC_IP>:8501"

    # Use a separate database file for production.
    DATABASE_URL="sqlite+aiosqlite:///./data/eido_sentinel_prod.db"

    # --- LLM & Geocoding Keys (REQUIRED) ---
    # Add your actual API key for the LLM provider.
    GOOGLE_API_KEY="your_real_google_api_key"

    # IMPORTANT: Provide a unique and real contact email in the user agent.
    GEOCODING_USER_AGENT="EidoSentinelApp/1.0 (contact: your-email@your-domain.com)"
    ```

    Save and exit the editor (for `nano`, press `Ctrl+X`, then `Y`, then `Enter`).

### Step 4: Run the Application with Docker Compose

With your production `.env` file ready, you can now start the application in detached mode.

```bash
sudo docker-compose up --build -d
```

- `--build`: Rebuilds the image if you've pulled new code changes.
- `-d`: Runs the containers in detached mode (in the background).

### Step 5: Access and Manage Your Application

- **Access the UI:** Open your browser and navigate to `http://<YOUR_SERVER_PUBLIC_IP>:8501`.
- **Access the API Docs:** `http://<YOUR_SERVER_PUBLIC_IP>:8000/docs`.

- **Check Logs:**
  To see the logs for the running services:

  ```bash
  # View logs for both services
  sudo docker-compose logs -f

  # View logs for just the API
  sudo docker-compose logs -f api

  # View logs for just the UId
  sudo docker-compose logs -f ui
  ```

  (Press `Ctrl+C` to stop viewing logs).

- **Stopping the Application:**
  To stop and remove the running containers:

  ```bash
  sudo docker-compose down
  ```

- **Updating the Application:**
  If you push new code to your repository:
  ```bash
  cd eido-sentinel
  git pull
  sudo docker-compose up --build -d
  ```
