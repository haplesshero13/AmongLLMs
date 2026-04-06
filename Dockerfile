FROM python:3.12-slim                                  
   
WORKDIR /app                                           
                
# Install uv                                           
RUN pip install uv
                                                        
COPY . .                                               

# Install dependencies with uv                         
RUN uv sync     
                                                        
EXPOSE 8080     

CMD ["sh", "-c", "cd human_trials && uv run uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}"]