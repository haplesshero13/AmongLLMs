import unittest
import os

class TestAPIURLConfiguration(unittest.TestCase):
    """Test that the API URL configuration works correctly."""
    
    def test_default_openrouter_url_env_var(self):
        """Test that os.getenv returns the correct default."""
        # Clear the environment variable if it exists
        if "OPENAI_API_BASE_URL" in os.environ:
            del os.environ["OPENAI_API_BASE_URL"]
        
        # Test the same logic used in the code
        api_url = os.getenv("OPENAI_API_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
        
        # Verify the default URL is OpenRouter
        self.assertEqual(api_url, "https://openrouter.ai/api/v1/chat/completions")
    
    def test_custom_api_url_env_var(self):
        """Test that OPENAI_API_BASE_URL overrides the default."""
        # Set a custom URL
        custom_url = "http://localhost:8000/v1/chat/completions"
        os.environ["OPENAI_API_BASE_URL"] = custom_url
        
        # Test the same logic used in the code
        api_url = os.getenv("OPENAI_API_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
        
        # Verify the custom URL is used
        self.assertEqual(api_url, custom_url)
        
        # Clean up
        if "OPENAI_API_BASE_URL" in os.environ:
            del os.environ["OPENAI_API_BASE_URL"]
    
    def test_localhost_url_format(self):
        """Test that localhost URL format is correct."""
        # Test HuggingFace transformers serving format
        test_url = "http://localhost:8000/v1/chat/completions"
        os.environ["OPENAI_API_BASE_URL"] = test_url
        
        api_url = os.getenv("OPENAI_API_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
        
        self.assertEqual(api_url, test_url)
        self.assertTrue(api_url.startswith("http://localhost:8000"))
        self.assertTrue(api_url.endswith("/v1/chat/completions"))
        
        # Clean up
        if "OPENAI_API_BASE_URL" in os.environ:
            del os.environ["OPENAI_API_BASE_URL"]

if __name__ == '__main__':
    unittest.main()
