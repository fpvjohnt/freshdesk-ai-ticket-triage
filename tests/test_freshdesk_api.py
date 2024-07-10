import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import pytz
import logging
from src.data.freshdesk import FreshdeskAPI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestFreshdeskAPI(unittest.TestCase):

    def setUp(self):
        self.api = FreshdeskAPI()
        logger.info("Set up TestFreshdeskAPI")

    def tearDown(self):
        logger.info("Tear down TestFreshdeskAPI")

    @patch('src.data.freshdesk.requests.request')
    def test_fetch_new_tickets(self, mock_request):
        logger.info("Testing fetch_new_tickets method")
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {'id': 1, 'created_at': '2024-01-01T00:00:00Z'},
            {'id': 2, 'created_at': '2024-01-02T00:00:00Z'}
        ]
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        start_date = datetime(2024, 1, 1, tzinfo=pytz.UTC)
        end_date = datetime(2024, 1, 3, tzinfo=pytz.UTC)

        try:
            tickets = self.api.fetch_new_tickets(start_date, end_date)
            self.assertEqual(len(tickets), 2)
            mock_request.assert_called()
            logger.info("fetch_new_tickets test passed")
        except Exception as e:
            logger.error(f"fetch_new_tickets test failed: {str(e)}")
            raise

    @patch('src.data.freshdesk.requests.request')
    def test_get_ticket_details(self, mock_request):
        logger.info("Testing get_ticket_details method")
        mock_response = MagicMock()
        mock_response.json.return_value = {'id': 1, 'subject': 'Test Ticket'}
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        try:
            ticket = self.api.get_ticket_details(1)
            self.assertEqual(ticket['id'], 1)
            self.assertEqual(ticket['subject'], 'Test Ticket')
            logger.info("get_ticket_details test passed")
        except Exception as e:
            logger.error(f"get_ticket_details test failed: {str(e)}")
            raise

    @patch('src.data.freshdesk.requests.request')
    def test_update_ticket_tags(self, mock_request):
        logger.info("Testing update_ticket_tags method")
        mock_response = MagicMock()
        mock_response.json.return_value = {'id': 1, 'tags': ['new_tag']}
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        try:
            updated_ticket = self.api.update_ticket_tags(1, ['new_tag'])
            self.assertEqual(updated_ticket['tags'], ['new_tag'])
            logger.info("update_ticket_tags test passed")
        except Exception as e:
            logger.error(f"update_ticket_tags test failed: {str(e)}")
            raise

    @patch('src.data.freshdesk.requests.request')
    def test_create_ticket(self, mock_request):
        logger.info("Testing create_ticket method")
        mock_response = MagicMock()
        mock_response.json.return_value = {'id': 1, 'subject': 'New Ticket', 'description': 'Test description'}
        mock_response.status_code = 201
        mock_request.return_value = mock_response

        try:
            new_ticket = self.api.create_ticket('New Ticket', 'Test description', 'test@example.com')
            self.assertEqual(new_ticket['subject'], 'New Ticket')
            self.assertEqual(new_ticket['description'], 'Test description')
            logger.info("create_ticket test passed")
        except Exception as e:
            logger.error(f"create_ticket test failed: {str(e)}")
            raise

    @patch('src.data.freshdesk.requests.request')
    def test_get_ticket_fields(self, mock_request):
        logger.info("Testing get_ticket_fields method")
        mock_response = MagicMock()
        mock_response.json.return_value = [{'id': 1, 'name': 'Priority'}, {'id': 2, 'name': 'Status'}]
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        try:
            fields = self.api.get_ticket_fields()
            self.assertEqual(len(fields), 2)
            self.assertEqual(fields[0]['name'], 'Priority')
            self.assertEqual(fields[1]['name'], 'Status')
            logger.info("get_ticket_fields test passed")
        except Exception as e:
            logger.error(f"get_ticket_fields test failed: {str(e)}")
            raise

    def test_api_error_handling(self):
        logger.info("Testing API error handling")
        with patch('src.data.freshdesk.requests.request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.json.return_value = {'errors': ['Bad Request']}
            mock_request.return_value = mock_response

            with self.assertRaises(Exception):
                self.api.get_ticket_details(1)
            logger.info("API error handling test passed")

if __name__ == '__main__':
    unittest.main()