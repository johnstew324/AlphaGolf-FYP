import aiohttp
import logging
from abc import ABC, abstractmethod
import asyncio
import time

class BaseDataScraper(ABC):
    def __init__(self, url, headers):
        self.url = url
        self.headers = headers
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = None
        self.last_request_time = 0
        self.min_request_interval = 1.0
    
    async def _get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def _wait_for_rate_limit(self):
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last_request)
        
        self.last_request_time = time.time()
    
    async def make_request(self, query, variables):
        await self._wait_for_rate_limit()
        
        try:
            session = await self._get_session()
            payload = {"query": query, "variables": variables}
            
            async with session.post(self.url, json=payload, headers=self.headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API failed: {response.status} - {error_text}")
                    
                return await response.json()
                
        except aiohttp.ClientError as e:
            self.logger.error(f"Request failed: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            raise
    
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    @abstractmethod
    async def parse_response(self, response_data):
        pass
