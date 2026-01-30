/**
 * Error Handler Utility
 * Sanitizes API error messages to prevent exposing internal system details
 * 
 * Security Issue: #47
 * https://github.com/INCF/knowledge-space-agent/issues/47
 */

interface ErrorPattern {
  pattern: RegExp;
  userMessage: string;
}

const ERROR_PATTERNS: ErrorPattern[] = [
  // Rate Limit Errors
  {
    pattern: /429|RESOURCE_EXHAUSTED|quota|rate.?limit/i,
    userMessage: "I'm currently experiencing high demand. Please try again in a moment."
  },
  // SSL/Connection Errors
  {
    pattern: /SSL|EOF|UNEXPECTED_EOF|protocol|_ssl\.c|ECONNREFUSED|ECONNRESET/i,
    userMessage: "Unable to connect to the server. Please check your connection and try again."
  },
  // Network Errors
  {
    pattern: /network|connection|fetch|failed.?to.?fetch|net::ERR/i,
    userMessage: "Unable to connect to the server. Please check your internet connection."
  },
  // Timeout Errors
  {
    pattern: /timeout|timed?\s*out|504|ETIMEDOUT/i,
    userMessage: "The request took too long. Please try a simpler query."
  },
  // Service Unavailable
  {
    pattern: /503|service.?unavailable|temporarily|maintenance/i,
    userMessage: "The service is temporarily unavailable. Please try again later."
  },
  // Server Errors
  {
    pattern: /500|internal.?server|server.?error/i,
    userMessage: "Something went wrong on our end. Please try again."
  },
  // Bad Gateway
  {
    pattern: /502|bad.?gateway/i,
    userMessage: "The service is temporarily unavailable. Please try again."
  },
  // Authentication Errors
  {
    pattern: /401|403|unauthorized|forbidden|auth/i,
    userMessage: "There was an authentication issue. Please try again later."
  }
];

/**
 * Patterns that indicate sensitive internal information
 */
const SENSITIVE_PATTERNS: RegExp[] = [
  /gemini|gpt-|claude|llama|anthropic|openai/i,
  /api[_-]?key|token|secret|password/i,
  /FreeTier|quota|billing|plan/i,
  /stack\s*trace|traceback|at\s+\w+\s+\(/i,
  /\/api\/|localhost|127\.0\.0\.1|0\.0\.0\.0/i,
  /\.py|\.js|\.ts|\.c|line\s*\d+|:\d+:\d+/i,
  /"error"\s*:\s*\{|\{.*"code"\s*:/i,
  /retryDelay|quotaValue|quotaId|quotaMetric/i,
  /_ssl\.c|ssl\.py|socket\.py/i,
  /violation.?of.?protocol|certificate/i
];

/**
 * Checks if a message contains sensitive internal details
 */
function containsSensitiveInfo(message: string): boolean {
  if (!message) return false;
  return SENSITIVE_PATTERNS.some(pattern => pattern.test(message));
}

/**
 * Sanitizes error messages to show user-friendly text
 * Prevents exposure of internal system details
 */
export function sanitizeErrorMessage(rawMessage: string): string {
  if (!rawMessage || typeof rawMessage !== 'string') {
    return "Something went wrong. Please try again.";
  }

  const message = rawMessage.trim();

  // Check against known error patterns first
  for (const { pattern, userMessage } of ERROR_PATTERNS) {
    if (pattern.test(message)) {
      return userMessage;
    }
  }

  // If contains any sensitive info, return generic message
  if (containsSensitiveInfo(message)) {
    return "Sorry, an error occurred while processing your request. Please try again.";
  }

  // If message is very long, it's probably a raw error dump
  if (message.length > 200) {
    return "Sorry, an error occurred. Please try again.";
  }

  // If message starts with "Error:" it's likely technical
  if (/^error\s*:/i.test(message)) {
    return "Sorry, an error occurred. Please try again.";
  }

  // Message seems safe, return it
  return message.length > 150 ? message.substring(0, 150) + '...' : message;
}

/**
 * Checks if a response indicates an error condition
 */
export function isErrorResponse(response: string): boolean {
  if (!response || typeof response !== 'string') return false;
  
  const errorIndicators = [
    /^error\s*:/i,
    /exception|failed|failure/i,
    /\b(4\d{2}|5\d{2})\b/,
    /RESOURCE_EXHAUSTED/i,
    /SSL.*error|EOF.*protocol/i,
    /unable\s+to|cannot\s+connect/i
  ];

  return errorIndicators.some(pattern => pattern.test(response));
}

/**
 * Logs error details for debugging (console only)
 */
export function logError(context: string, error: unknown): void {
  console.error(`[KnowledgeSpace Error] ${context}:`, error);
}