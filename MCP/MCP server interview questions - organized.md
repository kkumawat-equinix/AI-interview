# MCP Server Interview Questions & Answers

## 🎯 **1. Core Concepts & Fundamentals**

### Q1: What is Model Context Protocol (MCP) and why was it introduced?
**A:** MCP is a standardized protocol that enables AI applications to securely connect with external tools, resources, and data sources. It solves the problem of fragmented integrations by providing a unified interface for AI models to access external capabilities.

### Q2: Explain the three MCP primitives: Tools, Resources, and Prompts.
**A:** 
- **Tools**: Functions AI can call (e.g., `get_weather`, `send_email`)
- **Resources**: Static data sources AI can read (e.g., documentation, configs)  
- **Prompts**: Pre-defined templates to guide AI workflows

### Q3: How does MCP differ from OpenAI function calling or LangChain tools?
**A:** MCP provides standardized discovery, security, and transport layers. Unlike OpenAI's ad-hoc function calling, MCP offers structured capability discovery, multiple transport options, and built-in security features.

### Q4: What is JSON-RPC 2.0 and why is it a good fit for MCP?
**A:** JSON-RPC 2.0 is a lightweight remote procedure call protocol. It's ideal for MCP because it's stateless, has clear request/response structure, supports batching, and provides standardized error handling.

### Q5: What are the typical roles in MCP: Host, Client, and Server? How do they relate?
**A:** 
- **Host**: AI application (Claude, ChatGPT) that needs external capabilities
- **Client**: MCP client library that manages communication protocols
- **Server**: Exposes tools/resources via MCP protocol
- Flow: Host requests → Client discovers/calls → Server executes → Response flows back

---

## 🏗️ **2. Architecture & Lifecycle**

### Q6: Describe the architecture of an MCP system (Host → Client → Server).
**A:** Host (AI app) communicates through Client (MCP library) to Server (tool provider). Client handles protocol details, discovery, validation. Server exposes capabilities via standardized endpoints. Multiple servers can connect to one client.

### Q7: Walk through the lifecycle of a tool invocation in MCP.
**A:** 
1. Client calls `tools/list` for discovery
2. Host selects appropriate tool based on context
3. Client validates input against JSON schema
4. Client sends `tools/call` with validated parameters
5. Server executes tool logic and calls downstream APIs
6. Server returns structured response
7. Client forwards result to host

### Q8: What happens during capability discovery (`tools/list`, `resources/list`)?
**A:** Client queries server endpoints to get available capabilities with their schemas, descriptions, and requirements. This allows the host to understand what actions are possible before execution.

### Q9: Draw and explain the Host → MCP Client → MCP Server → Downstream API architecture.
**A:** Host sends requests to MCP Client, which discovers and validates against MCP Server capabilities. Server processes requests and calls Downstream APIs (databases, web services), then returns structured responses back through the chain.

### Q10: How do Resources differ from Tools in terms of read-only context and URIs?
**A:** Resources are read-only data accessed via URIs (like files), while Tools are executable functions. Resources provide context/documentation, Tools perform actions. Resources use URI-based addressing, Tools use name-based calling.

### Q11: What is the role of Prompts in MCP and how do they guide workflows?
**A:** Prompts provide pre-defined templates with parameters for complex workflows. They guide AI behavior by offering structured starting points, reducing prompt engineering needs, and ensuring consistent interactions.

### Q12: How do you version MCP servers or capabilities without breaking clients?
**A:** Use semantic versioning, maintain backward compatibility, implement capability negotiation, provide migration guides, and use feature flags for gradual rollouts.

### Q13: What conventions do you use for naming tools, resources, and prompts?
**A:** Use descriptive, verb-based names for tools (`get_weather`, `send_email`), noun-based URIs for resources (`/docs/api.md`), and workflow-based names for prompts (`code_review_template`). Follow consistent naming patterns across domains.

---

## 🚀 **3. Transport & Connectivity** 

### Q14: Which transports does MCP support and when do you use each?
**A:** 
- **stdio**: Local development, desktop apps (direct process communication)
- **Streamable HTTP**: Production, remote servers (HTTP with streaming)
- **WebSocket**: Real-time applications (bidirectional communication)

### Q15: Why is Streamable HTTP preferred over SSE for remote servers?
**A:** Streamable HTTP supports bidirectional communication, better error handling, proper request/response correlation, and full HTTP semantics. SSE is unidirectional and lacks structured request handling.

### Q16: How does stdio transport work and why is it useful for local development?
**A:** Uses standard input/output streams for communication. Process spawned with JSON-RPC messages sent via stdin and responses received via stdout. Ideal for local tools, desktop integration, and development testing.

### Q17: What are the pitfalls of pure SSE vs Streamable HTTP in MCP?
**A:** SSE lacks bidirectional communication, proper error propagation, request correlation, and connection management. Streamable HTTP provides full duplex communication with better reliability and debugging capabilities.

### Q18: How do you implement streaming progress/partial results over Streamable HTTP?
**A:** Use chunked transfer encoding with JSON-RPC notifications for progress updates, implement proper connection handling, buffer management, and provide cancellation mechanisms for long-running operations.

---

## 🔒 **4. Security & Authentication**

### Q19: How do you secure MCP servers exposed over HTTP?
**A:** 
- TLS encryption for all communications
- Origin validation to prevent unauthorized access
- Authentication tokens (Bearer, OAuth2)
- Network boundaries (firewalls, VPC)
- Input validation and sanitization
- Rate limiting and DDoS protection

### Q20: What is Origin validation and why is it important?
**A:** Validates the requesting client's origin header to prevent unauthorized access. Critical for preventing cross-site request forgery and ensuring only approved clients can access the MCP server.

### Q21: How would you implement OAuth2 or bearer token authentication for MCP?
**A:** Use Client Credentials flow: acquire token from auth server, cache with expiry handling, include in Authorization header, implement token refresh logic, and handle authentication failures gracefully.

### Q22: What measures prevent unauthorized tool invocation or data leakage?
**A:** RBAC (role-based access control), tool allowlists/denylists, user consent flows for sensitive operations, input validation, audit logging, and principle of least privilege access.

### Q23: Describe OAuth2 Client Credentials integration for downstream APIs.
**A:** Acquire access token using client_credentials grant, cache tokens with proper expiry handling, include in Authorization headers for downstream calls, implement token refresh before expiry, and handle auth errors gracefully.

### Q24: What is RBAC for MCP tools and how would you implement allowlists/denylists?
**A:** Role-Based Access Control restricts tool access based on user roles. Implement using middleware that checks user permissions against tool requirements, maintain role-to-tool mappings, and provide admin interfaces for management.

### Q25: How do you enforce user consent for sensitive or side-effectful tools?
**A:** Implement consent flows in the client UI, maintain consent records, require explicit approval for dangerous operations, provide clear descriptions of tool effects, and allow consent revocation.

### Q26: What types of input validation and schema hardening do you apply?
**A:** Use strict JSON schemas with enums, patterns, min/max values, required fields, additionalProperties:false, input sanitization, and type coercion prevention to ensure data integrity.

### Q27: How do you prevent prompt/tool injection or command injection via tool inputs?
**A:** Sanitize all inputs, use parameterized queries, avoid direct command execution, implement input whitelisting, escape special characters, and validate against strict schemas.

### Q28: What's your approach to secrets management?
**A:** Use environment variables for development, implement proper secret rotation, leverage cloud secret managers (AWS Secrets Manager, Azure Key Vault), never log sensitive data, and use encrypted storage.

### Q29: How do you handle multi-tenant or per-customer isolation in a shared MCP server?
**A:** Implement tenant-scoped authentication, separate database schemas/namespaces, isolate resource access, use tenant-specific configuration, and ensure audit trails include tenant context.

---

## 📋 **5. Tool Design & JSON Schema**

### Q30: What makes a "good" tool schema? Provide examples of tight vs loose schemas.
**A:** Good schemas are specific, well-documented, and validated. Tight: `{location: {enum: ["NYC", "LA"]}}`. Loose: `{location: {type: "string"}}`. Balance specificity with flexibility based on use case requirements.

### Q31: How do you validate tool inputs and prevent injection attacks?
**A:** Use strict JSON Schema validation with patterns, min/max values, enums for constrained inputs, sanitize string inputs, validate against whitelists, and never directly execute user input.

### Q32: Why is strict JSON Schema validation critical in MCP?
**A:** Prevents injection attacks, ensures data integrity, provides clear error messages, enables proper type checking, maintains API contract compliance, and improves debugging capabilities.

### Q33: How do you design coarse-grained tools to reduce context bloat and round trips?
**A:** Combine related operations into single tools, accept complex objects as parameters, return comprehensive responses, batch similar operations, and design workflow-oriented rather than CRUD-oriented tools.

### Q34: What strategies do you use to ensure idempotency for create/update operations?
**A:** Use idempotency keys, implement upsert operations, check for existing resources before creating, return consistent responses for duplicate requests, and design stateless operations where possible.

### Q35: How do you handle long-running tasks—streaming progress, checkpoints, cancelation?
**A:** Implement streaming responses for progress updates, use task IDs for status polling, provide cancellation endpoints, implement checkpointing for recovery, and return partial results when appropriate.

### Q36: How do you structure tool outputs for model consumption?
**A:** Use consistent field names, provide clear success/error indicators, include human-readable messages, structure data hierarchically, and include relevant metadata for decision-making.

---

## ⚡ **6. Performance & Scalability**

### Q37: How do you scale MCP when you have 50–100 tools without exploding the context window?
**A:** Group tools by domain, implement lazy loading, use pagination for tool lists, design coarse-grained operations, cache tool metadata, and implement smart tool selection based on context.

### Q38: What strategies reduce context bloat and latency?
**A:** Minimize tool descriptions, use efficient schemas, cache static data, batch related operations, implement smart tool selection, and provide only relevant capabilities based on context.

### Q39: How do you handle multiple downstream API calls inside a single tool?
**A:** Use concurrent execution where possible, implement circuit breakers, apply timeouts, use connection pooling, batch related calls, and handle partial failures gracefully.

### Q40: What are your caching strategies for static lookups?
**A:** Implement in-memory caching for frequently accessed data, use Redis for shared caching, implement cache invalidation strategies, cache tool metadata and schemas, and use CDNs for static resources.

### Q41: When do you batch internal downstream API calls vs parallelize them?
**A:** Batch when APIs support it and data is related; parallelize for independent operations. Consider rate limits, latency requirements, error handling complexity, and resource utilization patterns.

### Q42: How do you design pagination for tools/list or resource listings?
**A:** Use cursor-based pagination for consistency, implement reasonable page sizes, provide total counts when possible, support filtering and sorting, and ensure stable ordering.

### Q43: What metrics do you track?
**A:** Request latency (p50/p95/p99), error rates by tool, throughput, downstream API response times, authentication success rates, resource utilization, and cache hit rates.

### Q44: How do you apply rate limiting, backoff/retries, and circuit breakers?
**A:** Implement token bucket rate limiting, exponential backoff with jitter, circuit breakers for downstream services, different retry strategies based on error types, and proper timeout handling.

---

## 📊 **7. Observability & Reliability**

### Q45: What do you log for each MCP request?
**A:** Request ID, method name, tool name, user context, input parameters (sanitized), response status, execution time, downstream API calls, and error details for comprehensive debugging and audit trails.

### Q46: How do you implement retries/backoff for downstream failures?
**A:** Exponential backoff with jitter, maximum retry limits, circuit breaker patterns, different strategies based on error types (network vs application), and proper timeout handling.

### Q47: How do you support long-running tasks in MCP?
**A:** Implement streaming responses, provide progress updates, support task cancellation, use checkpointing for recovery, return task IDs for status polling, and handle connection drops gracefully.

### Q48: How do you propagate correlation IDs across MCP and downstream calls?
**A:** Generate correlation IDs at request entry, pass through all downstream calls via headers, include in all log entries, use for distributed tracing, and return in error responses.

### Q49: How do you implement structured logging and OpenTelemetry tracing?
**A:** Use structured JSON logging with consistent fields, implement distributed tracing with spans, correlate logs and traces, monitor critical paths, and integrate with observability platforms.

### Q50: What's your approach to alerting and runbooks for MCP incidents?
**A:** Monitor key metrics (error rates, latency), create actionable alerts with clear escalation paths, maintain runbooks for common issues, implement automated remediation where possible, and conduct post-incident reviews.

### Q51: How do you design error payloads—codes, messages, retry hints, partial results?
**A:** Use standardized error codes, provide human-readable messages, include retry guidance, return partial results when useful, and structure errors consistently across all tools.

---

## 💻 **8. Practical & Coding**

### Q52: Show how to register a tool in Node.js using the MCP SDK.
**A:** 
```typescript
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [{
    name: "get_weather",
    description: "Get weather for a location",
    inputSchema: {
      type: "object",
      properties: {
        location: { type: "string" }
      },
      required: ["location"]
    }
  }]
}));
```

### Q53: Show how to expose a FastAPI endpoint as an MCP tool.
**A:**
```python
@app.post("/mcp/tools/call")
async def call_tool(request: ToolCallRequest):
    if request.params.name == "get_weather":
        location = request.params.arguments["location"]
        weather_data = await fetch_weather(location)
        return {"content": [{"type": "text", "text": str(weather_data)}]}
```

### Q54: How do you deploy an MCP server to production securely?
**A:** Use containers with minimal base images, implement TLS termination, add authentication middleware, configure proper networking (VPC), implement health checks, use secrets management, and set up monitoring/logging.

---

## 🚀 **9. Advanced Topics**

### Q55: How do you version tools and resources in MCP?
**A:** Use semantic versioning in tool names/namespaces, maintain backward compatibility, implement feature flags, provide migration guides, use API versioning headers, and gradual deprecation strategies.

### Q56: How do you design prompts for complex workflows?
**A:** Create modular prompts with clear parameters, use templates for consistency, provide examples, chain prompts for multi-step processes, include error handling guidance, and test with various inputs.

### Q57: What is RBAC in MCP and how would you implement it?
**A:** Role-Based Access Control restricts tool access based on user roles. Implement using middleware that checks user permissions against tool requirements, maintain role-to-tool mappings, and provide admin interfaces.

### Q58: How do you test MCP servers (unit, integration, manual)?
**A:** 
- **Unit tests**: Individual tool handlers with mocked dependencies
- **Integration tests**: Full MCP protocol flow with test clients  
- **Contract tests**: Schema validation and API compliance
- **Manual testing**: MCP Inspector and client applications

### Q59: What's new in the latest MCP spec update (e.g., Streamable HTTP)?
**A:** Latest updates include Streamable HTTP transport for better bidirectional communication, enhanced security features, improved error handling, resource subscription capabilities, and standardized authentication patterns.

---

## 🛠️ **10. Resources & Implementation Details**

### Q60: How do you design resource URIs (namespacing, versioning, mutable vs immutable)?
**A:** Use hierarchical namespacing (`/api/v1/docs/guide.md`), implement versioning in URIs, distinguish mutable vs immutable resources, provide clear resource lifecycle management, and support content negotiation.

### Q61: When do you prefer Resources over Tools for providing documentation/config?
**A:** Use Resources for static content, documentation, configuration files, schemas, and reference data. Use Tools for dynamic operations, computations, API calls, and state-changing actions.

### Q62: How do you design Prompts for complex workflows?
**A:** Structure with clear parameters, provide contextual examples, support template variables, enable prompt chaining, include success criteria, and design for reusability across scenarios.

### Q63: How do you keep prompts concise but task-specific for high model accuracy?
**A:** Focus on essential instructions, use clear action verbs, provide specific examples, eliminate ambiguity, structure information hierarchically, and test with target models.

---

## 🏢 **11. Client/Host Behavior & Team Practices**

### Q64: How do MCP clients typically discover and select tools?
**A:** Clients call discovery endpoints, cache tool metadata, match tools to user intent using semantic analysis, consider tool dependencies, and provide fallback options when primary tools are unavailable.

### Q65: What's your approach to client consent UIs for high-risk tools?
**A:** Implement clear warning messages, require explicit confirmation, show tool effects preview, maintain consent history, allow granular permissions, and provide easy revocation mechanisms.

### Q66: How do you handle client batching for discovery vs single tool calls for execution?
**A:** Batch discovery calls to reduce overhead, cache capabilities locally, use single calls for tool execution to maintain request tracing, and implement efficient connection management.

### Q67: How do you keep the client context small while giving enough capabilities?
**A:** Implement lazy loading of tools, use context-aware tool selection, cache only essential metadata, provide tool grouping, and support dynamic capability discovery.

### Q68: Monolith vs multi-domain MCP servers—trade-offs and migration strategy.
**A:** 
- **Monolith**: Easier deployment, shared resources, simpler auth
- **Multi-domain**: Better isolation, independent scaling, team ownership
- **Migration**: Start monolith, split by domain boundaries, implement service mesh

### Q69: How do you split tool sets across teams or services?
**A:** Organize by business domain, maintain clear ownership boundaries, implement shared infrastructure, use common authentication, and provide cross-team collaboration tools.

### Q70: How do you ensure backwards compatibility when evolving schemas?
**A:** Version your schemas, avoid breaking changes, provide migration guides, use feature flags for gradual rollouts, maintain multiple schema versions, and implement deprecation timelines.

---

## 🔮 **12. Advanced & Future-Looking**

### Q71: How do you design multi-step workflows (tools + prompts + resources)?
**A:** Chain prompts for guidance, use tools for actions, leverage resources for context, implement workflow state management, provide checkpoints, and enable workflow resumption.

### Q72: What are the security hardening steps unique to MCP?
**A:** Origin validation, capability-based security, tool consent mechanisms, input sanitization, audit logging, principle of least privilege, and secure transport requirements.

### Q73: How would you expose customer-defined tools safely in a shared server?
**A:** Implement sandboxing, use strict resource limits, validate custom tool schemas, provide isolation boundaries, implement approval workflows, and monitor for malicious behavior.

### Q74: What would you change in your MCP design if downstream APIs have strict rate limits?
**A:** Implement request queuing, cache aggressively, batch operations, use circuit breakers, implement priority systems, and provide rate limit feedback to clients.

### Q75: How do you plan for global deployments (latency, data residency, failover)?
**A:** Use regional deployments, implement data locality, provide failover mechanisms, optimize for latency, ensure compliance with data regulations, and implement global load balancing.

### Q76: Which MCP SDK/language would you pick for your org and why?
**A:** Consider team expertise, performance requirements, ecosystem maturity, library support, deployment infrastructure, and long-term maintenance needs. TypeScript/Node.js offers good balance of performance and developer experience.

### Q77: How do you handle schema/version negotiation between client and server?
**A:** Implement capability negotiation during handshake, support multiple schema versions, use feature detection, provide graceful degradation, and maintain compatibility matrices.

### Q78: How do you approach auditability for compliance?
**A:** Log all tool invocations with user context, maintain immutable audit trails, implement retention policies, provide audit reports, ensure data integrity, and support compliance reporting requirements.

---

## 📚 **Interview Success Tips**

### 🎯 **Preparation Strategy**
- **Study the MCP specification** thoroughly
- **Build sample MCP servers** in different languages
- **Practice explaining concepts** simply and clearly
- **Prepare real-world examples** from your experience
- **Understand trade-offs** between different approaches

### 💡 **During the Interview**
- **Start with fundamentals** before diving into details
- **Use diagrams** to explain architecture concepts
- **Provide specific examples** rather than abstract explanations
- **Discuss trade-offs** and decision-making processes
- **Ask clarifying questions** to understand requirements

### 🚀 **Advanced Preparation**
- **Know the latest MCP updates** and roadmap
- **Understand integration patterns** with popular AI platforms
- **Study performance optimization** techniques
- **Practice system design** for MCP-based architectures
- **Review security best practices** for AI tool integrations