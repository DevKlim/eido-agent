# EIDO Sentinel

EIDO Sentinel is an AI-powered platform designed to enhance emergency response by intelligently processing, correlating, and analyzing diverse emergency data streams.

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
    **IMPORTANT:** You must edit the new `.env` file to provide your API keys (e.g., `GOOGLE_API_KEY`) and a unique `GEOCODING_USER_AGENT` with your email. For local use, the default URLs are fine.

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

## Deployment to a Cloud VM (e.g., Oracle Cloud Free Tier)

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

-   **Oracle Cloud (VCN Security List):**
    1.  In your OCI console, navigate to your Virtual Cloud Network (VCN).
    2.  Go to "Security Lists" and select the one associated with your VM's subnet.
    3.  Click "Add Ingress Rules".
    4.  Create a rule:
        -   **Source CIDR:** `0.0.0.0/0` (allows traffic from any IP)
        -   **IP Protocol:** TCP
        -   **Destination Port Range:** `8000,8501`
    5.  Add the rule.

-   **On the VM's Firewall (if active):**
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

-   **Access the UI:** Open your browser and navigate to `http://<YOUR_SERVER_PUBLIC_IP>:8501`.
-   **Access the API Docs:** `http://<YOUR_SERVER_PUBLIC_IP>:8000/docs`.

-   **Check Logs:**
    To see the logs for the running services:
    ```bash
    # View logs for both services
    sudo docker-compose logs -f

    # View logs for just the API
    sudo docker-compose logs -f api

    # View logs for just the UI
    sudo docker-compose logs -f ui
    ```
    (Press `Ctrl+C` to stop viewing logs).

-   **Stopping the Application:**
    To stop and remove the running containers:
    ```bash
    sudo docker-compose down
    ```

-   **Updating the Application:**
    If you push new code to your repository:
    ```bash
    cd eido-sentinel
    git pull                      # Get the latest code
    sudo docker-compose up --build -d # Rebuild and restart with the new code