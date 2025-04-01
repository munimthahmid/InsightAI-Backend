    def _post_process_data(self, data, query):
        """Post-process web search results."""
        results = []
        for item in data:
            # Extract the main content
            title = item.get("title", "Untitled Web Page")
            url = item.get("link", "")
            snippet = item.get("snippet", "")
            
            # Create a meaningful metadata structure
            metadata = {
                "title": title,
                "url": url,
                "source_type": "web",
                "query": query,
            }
            
            # Ensure the URL is present in both the metadata and directly
            # in the document for better visibility to LLM
            content = f"Title: {title}\nURL: {url}\n\n{snippet}"
            
            # Create the final document
            results.append({"page_content": content, "metadata": metadata})
            
        return results 