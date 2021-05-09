import json
import os
import re
import scrapy
from scrapy.selector import Selector
import string

class DebateSpider(scrapy.Spider):
    name = "debate_crawler"
    start_urls = ["https://www.debate.org/opinions/?sort=popular"]

    # Constants
    TOPIC_LIMIT = 5
    BASE_URL = "https://www.debate.org"
    LOAD_MORE_ARGUMENTS = "https://www.debate.org/opinions/~services/opinions.asmx/GetDebateArgumentPage"
    LIST_DATA_HEADING_CLASS = "li.hasData h2"
    LIST_DATA_PARAGRAPH_CLASS = "li.hasData p"
    
    # Remove json file if it exists
    file_name = "data.json"
    if os.path.exists(file_name):
        os.remove(file_name)
    
    # Crawl the first page (debate.org main page) 
    def parse(self, response):
        debate_topics = response.css("a.a-image-contain::attr(href)").getall()[:self.TOPIC_LIMIT]
        debate_topics = [self.BASE_URL+ext for ext in debate_topics]
        return (scrapy.Request(url, callback=self.crawl_each_debate_topic) for url in debate_topics)

    # Create dictionary for argument title and body
    # This is common for both pro and con arguments
    def collect_all_arguments(self, titles_array, contents_array):
        return_array = []
        for i in range(len(titles_array)):
            return_array.append({
                'title': self.remove_html_tags(titles_array[i]),
                'body': self.remove_html_tags(contents_array[i])
            })
        return return_array

    # Extract pro and con arguments
    def crawl_pro_con_arguements_from_response(self, response):
        id_list = ["#yes-arguments","#no-arguments"]
        for i in id_list:
            titles = response.css(i+" "+self.LIST_DATA_HEADING_CLASS).getall()
            contents = response.css(i+" "+self.LIST_DATA_PARAGRAPH_CLASS).getall()
            if '#yes' in i:
                pro_arguments = self.collect_all_arguments(titles, contents)
            else:
                con_arguments = self.collect_all_arguments(titles, contents)
        return {
            'pro_arguments': pro_arguments,
            'con_arguments': con_arguments
        }

    # Extract category, title of the debate
    def extract_title_category_debateId(self, response):
        debate_category = response.css("#breadcrumb a:nth-child(3)::text").get()
        debate_title = response.css("#col-wi span.q-title::text").get()
        debate_guid = response.css(".hasData::attr(did)").get()
        return {
            'topic': debate_title,
            'category': debate_category,
            'debateId': debate_guid,
        }

    # Crawl the first page of every debate topic and
    # Get more arguments from 'Load More Arguments' button click 
    def crawl_each_debate_topic(self, response):
        crawled_data = self.extract_title_category_debateId(response)
        arguments_object = self.crawl_pro_con_arguements_from_response(response)
        crawled_data['pro_arguments'] = arguments_object['pro_arguments']
        crawled_data['con_arguments'] = arguments_object['con_arguments']
        yield from self.load_more_arguments(crawled_data, 2)
        
    # Make POST request to fetch more arguments from the server
    def load_more_arguments(self, crawled_data, page_number):
        headers = self.get_request_header()            
        data = {
            "debateId": crawled_data["debateId"],
            "pageNumber": page_number,
            "itemsPerPage": 10,
            "ysort": 5,
            "nsort": 5
        }
        yield scrapy.http.Request(
            self.LOAD_MORE_ARGUMENTS,
            method="POST",
            body=json.dumps(data),
            headers= headers,
            callback=self.crawl_more_arguments,
            cb_kwargs=dict(crawled_data = crawled_data, page = page_number)
            )
       
    # Crawl the response of more arguments and append them to existing arguments
    # {ddo.split} is the separator between pro-arguments, con-arguments and
    # if all data has been loaded, it returns 'finished' as keyword
    # if data is still pending, it return 'needmore' as keyword 
    def crawl_more_arguments(self, response, crawled_data, page):
        output = json.loads(response.text)
        output = output["d"]
        pro_section = Selector(text=output.split("{ddo.split}")[0])
        con_section = Selector(text=output.split("{ddo.split}")[1])
        continue_flag = False if 'finished' in output.split("{ddo.split}")[2] else True
        
        # Extract pro and con arguments from new resposne
        pro_arguments = self.collect_all_arguments(
            pro_section.css(self.LIST_DATA_HEADING_CLASS).getall(),
            pro_section.css(self.LIST_DATA_PARAGRAPH_CLASS).getall()
        )
        con_arguments = self.collect_all_arguments(
            con_section.css(self.LIST_DATA_HEADING_CLASS).getall(),
            con_section.css(self.LIST_DATA_PARAGRAPH_CLASS).getall()
        )
        
        # Append new arguments and existing arguments
        crawled_data["pro_arguments"] = crawled_data["pro_arguments"] + pro_arguments
        crawled_data["con_arguments"] = crawled_data["con_arguments"] + con_arguments
        
        # finished not found in response, get more arguments from POST request
        if continue_flag:
            yield from self.load_more_arguments(crawled_data, page+1)
        # finished found in response, return entire crawled data
        else:
            yield crawled_data

    # Get Headers for Load More Arguments POST request
    def get_request_header(self):
        return {
            "Content-Type": "application/json;charset=UTF-8",
            "Accept":"application/json, text/plain, */*"
            }

    # Remove html tags and return clean string for 
    # arguments titles and arguments body using regex 
    def remove_html_tags(self, html_text):
        remover_regex = re.compile('<.*?>')
        html_free_text = re.sub(remover_regex, '', html_text)
        return html_free_text