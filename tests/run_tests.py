import subprocess
import sys
import time
import threading

def start_api_server():
    """Start the API server in background"""
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "fastapi_backend:app", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ])

def run_all_tests():
    """Run all tests"""
    print("Starting Punjabi Folk Song Recommender Tests...\n")
    
    # Start API server
    print("Starting API server...")
    server_process = start_api_server()
    
    try:
        # Wait for server to start
        time.sleep(10)
        
        # Run integration tests
        print("Running integration tests...")
        tester = APIIntegrationTest()
        
        if tester.wait_for_api():
            tester.test_full_workflow()
            tester.performance_test()
            
            # Run demo
            print("\nRunning demo...")
            demo_client = PunjabiFolkSongClient()
            demo_client.run_full_demo()
        else:
            print("❌ API failed to start")
            
    finally:
        # Clean up
        print("\nStopping API server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    run_all_tests()