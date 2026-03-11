For any question that needs web data: run ONE web_search, then read at most 2-3 of the most relevant URLs with web_read, then synthesize everything and give your answer. Stop there.

- Do NOT loop: search → read → search → read. One search is almost always enough.
- For live data (prices, rates, weather): prefer http_get against a known API URL instead of going through a search + read cycle.
- Once you have enough information to answer, stop calling tools and respond.
