class FreshdeskAPI:
    def __init__(self):
        self.base_url = f"{FRESHDESK_DOMAIN}/api/v2"
        self.headers = {
            "Content-Type": "application/json",
        }

    def fetch_new_tickets(self, start_date, end_date):
        url = f"{self.base_url}/tickets"
        params = {
            "order_by": "created_at",
            "order_type": "desc",
            "per_page": 100,
            "page": 1
        }
        all_tickets = []
        
        logger.info(f"Fetching tickets from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        while True:
            response = requests.get(url, headers=self.headers, auth=(API_KEY, 'X'), params=params)
            if response.status_code == 200:
                tickets = response.json()
                logger.info(f"Retrieved {len(tickets)} tickets on page {params['page']}")
                if not tickets:
                    break
                all_tickets.extend(tickets)
                if len(tickets) < 100:
                    break
                params["page"] += 1
            else:
                logger.error(f"Failed to retrieve tickets. Status code: {response.status_code}")
                logger.error(f"Response content: {response.text}")
                break
        
        filtered_tickets = [
            ticket for ticket in all_tickets
            if start_date <= datetime.datetime.fromisoformat(ticket['created_at'].replace('Z', '+00:00')) < end_date
        ]
        
        logger.info(f"Total tickets found within the date range: {len(filtered_tickets)}")
        return filtered_tickets

    def get_conversation(self, ticket_id):
        url = f"{self.base_url}/tickets/{ticket_id}/conversations"
        response = requests.get(url, headers=self.headers, auth=(API_KEY, 'X'))
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to retrieve conversation for ticket {ticket_id}. Status code: {response.status_code}")
            return []

    def get_ticket_details(self, ticket_id):
        url = f"{self.base_url}/tickets/{ticket_id}"
        response = requests.get(url, headers=self.headers, auth=(API_KEY, 'X'))
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to retrieve details for ticket {ticket_id}. Status code: {response.status_code}")
            return None

    def update_ticket_tags(self, ticket_id, tags):
        url = f"{self.base_url}/tickets/{ticket_id}"
        data = {"tags": tags}
        response = requests.put(url, headers=self.headers, auth=(API_KEY, 'X'), json=data)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to update tags for ticket {ticket_id}. Status code: {response.status_code}")
            return None

    def create_ticket(self, subject, description, email):
        url = f"{self.base_url}/tickets"
        data = {
            "subject": subject,
            "description": description,
            "email": email,
            "status": 2,
            "priority": 1
        }
        response = requests.post(url, headers=self.headers, auth=(API_KEY, 'X'), json=data)
        if response.status_code == 201:
            return response.json()
        else:
            logger.error(f"Failed to create ticket. Status code: {response.status_code}")
            return None

    def get_ticket_fields(self):
        url = f"{self.base_url}/ticket_fields"
        response = requests.get(url, headers=self.headers, auth=(API_KEY, 'X'))
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to retrieve ticket fields. Status code: {response.status_code}")
            return None
