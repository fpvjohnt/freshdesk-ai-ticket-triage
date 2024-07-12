import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import pytz
import logging
import os
import sys

# Ensure the src/data directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'data')))

from freshdesk import FreshdeskAPI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestFreshdeskAPI(unittest.TestCase):

    @patch.dict(os.environ, {'FRESHDESK_DOMAIN': 'https://cintoo.freshdesk.com', 'FRESHDESK_API_KEY': 'AYrGLqYvCFrlwBTMEFb'})
    def setUp(self):
        self.api = FreshdeskAPI()
        logger.info("Set up TestFreshdeskAPI")

    def tearDown(self):
        logger.info("Tear down TestFreshdeskAPI")

    @patch('freshdesk.requests.get')
    def test_fetch_new_tickets(self, mock_get):
        logger.info("Testing fetch_new_tickets method")
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {'id': 1, 'created_at': '2024-01-01T00:00:00Z'},
            {'id': 2, 'created_at': '2024-01-02T00:00:00Z'}
        ]
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        start_date = datetime(2024, 1, 1, tzinfo=pytz.UTC)
        end_date = datetime(2024, 1, 3, tzinfo=pytz.UTC)

        tickets = self.api.fetch_new_tickets(start_date, end_date)
        self.assertEqual(len(tickets), 2)
        mock_get.assert_called()
        logger.info("fetch_new_tickets test passed")

    @patch('freshdesk.requests.get')
    def test_get_ticket_details(self, mock_get):
        logger.info("Testing get_ticket_details method")
        mock_response = MagicMock()
        mock_response.json.return_value = {'id': 1, 'subject': 'Test Ticket'}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        ticket = self.api.get_ticket_details(1)
        self.assertIsNotNone(ticket)
        self.assertEqual(ticket['id'], 1)
        self.assertEqual(ticket['subject'], 'Test Ticket')
        logger.info("get_ticket_details test passed")

    @patch('freshdesk.requests.put')
    def test_update_ticket_tags(self, mock_put):
        logger.info("Testing update_ticket_tags method")
        mock_response = MagicMock()
        mock_response.json.return_value = {'id': 1, 'tags': ['new_tag']}
        mock_response.status_code = 200
        mock_put.return_value = mock_response

        updated_ticket = self.api.update_ticket_tags(1, ['new_tag'])
        self.assertIsNotNone(updated_ticket)
        self.assertEqual(updated_ticket['tags'], ['new_tag'])
        logger.info("update_ticket_tags test passed")

    @patch('freshdesk.requests.post')
    def test_create_ticket(self, mock_post):
        logger.info("Testing create_ticket method")
        mock_response = MagicMock()
        mock_response.json.return_value = {'id': 1, 'subject': 'New Ticket', 'description': 'Test description'}
        mock_response.status_code = 201
        mock_post.return_value = mock_response

        new_ticket = self.api.create_ticket('New Ticket', 'Test description', 'test@example.com')
        self.assertIsNotNone(new_ticket)
        self.assertEqual(new_ticket['subject'], 'New Ticket')
        self.assertEqual(new_ticket['description'], 'Test description')
        logger.info("create_ticket test passed")

    @patch('freshdesk.requests.get')
    def test_get_ticket_fields(self, mock_get):
        logger.info("Testing get_ticket_fields method")
        mock_response = MagicMock()
        mock_response.json.return_value = [{'id': 1, 'name': 'Priority'}, {'id': 2, 'name': 'Status'}]
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        fields = self.api.get_ticket_fields()
        self.assertIsNotNone(fields)
        self.assertEqual(len(fields), 2)
        self.assertEqual(fields[0]['name'], 'Priority')
        self.assertEqual(fields[1]['name'], 'Status')
        logger.info("get_ticket_fields test passed")

    @patch.dict(os.environ, {'FRESHDESK_DOMAIN': 'https://cintoo.freshdesk.com', 'FRESHDESK_API_KEY': 'AYrGLqYvCFrlwBTMEFb'})
    def test_api_error_handling(self):
        logger.info("Testing API error handling")
        with patch('freshdesk.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response

            with self.assertRaises(Exception) as context:
                self.api.get_ticket_details(1)
            
            self.assertTrue('Ticket 1 not found' in str(context.exception))
            mock_get.assert_called_with(f'https://cintoo.freshdesk.com/api/v2/tickets/1', headers=self.api.headers, auth=('AYrGLqYvCFrlwBTMEFb', 'X'))
            logger.info("API error handling test passed")

if __name__ == '__main__':
    unittest.main()
