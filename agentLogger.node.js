/**
 * Agent Logger Module for Node.js
 * Server-side logging system for agent-based applications
 * Only logs: Active Agent, Agent I/O, Decisions, Tools, Models, and Prompts
 */

const fs = require('fs');
const path = require('path');
const util = require('util');

class AgentLogger {
    constructor(config = {}) {
        this.config = {
            enableActiveAgent: true,
            enableAgentIO: true,
            enableDecisions: true,
            enableTools: true,
            enableModels: true,
            enablePrompts: true,
            logLevel: config.logLevel || 'info',
            outputFormat: config.outputFormat || 'console', // console, file, both, or custom
            logFilePath: config.logFilePath || path.join(process.cwd(), 'agent.log'),
            maxFileSize: config.maxFileSize || 10 * 1024 * 1024, // 10MB default
            ...config
        };
        
        this.logHandlers = {
            console: this.logToConsole.bind(this),
            file: this.logToFile.bind(this),
            both: this.logToBoth.bind(this),
            custom: config.customHandler || this.logToConsole.bind(this)
        };

        // Ensure log directory exists
        if (this.config.outputFormat === 'file' || this.config.outputFormat === 'both') {
            const logDir = path.dirname(this.config.logFilePath);
            if (!fs.existsSync(logDir)) {
                fs.mkdirSync(logDir, { recursive: true });
            }
        }
    }

    /**
     * Log active agent information
     */
    logActiveAgent(agentName, agentId, metadata = {}) {
        if (!this.config.enableActiveAgent) return;
        
        this.log('ACTIVE_AGENT', {
            type: 'active_agent',
            agentName,
            agentId,
            timestamp: new Date().toISOString(),
            pid: process.pid,
            ...metadata
        });
    }

    /**
     * Log agent input
     */
    logAgentInput(agentName, input, metadata = {}) {
        if (!this.config.enableAgentIO) return;
        
        this.log('AGENT_INPUT', {
            type: 'agent_input',
            agentName,
            input: this.truncateIfNeeded(input),
            timestamp: new Date().toISOString(),
            ...metadata
        });
    }

    /**
     * Log agent output
     */
    logAgentOutput(agentName, output, metadata = {}) {
        if (!this.config.enableAgentIO) return;
        
        this.log('AGENT_OUTPUT', {
            type: 'agent_output',
            agentName,
            output: this.truncateIfNeeded(output),
            timestamp: new Date().toISOString(),
            ...metadata
        });
    }

    /**
     * Log agent decisions
     */
    logDecision(agentName, decision, reasoning = '', metadata = {}) {
        if (!this.config.enableDecisions) return;
        
        this.log('DECISION', {
            type: 'decision',
            agentName,
            decision,
            reasoning: this.truncateIfNeeded(reasoning),
            timestamp: new Date().toISOString(),
            ...metadata
        });
    }

    /**
     * Log tool usage
     */
    logToolUsage(agentName, toolName, toolParams = {}, result = null, metadata = {}) {
        if (!this.config.enableTools) return;
        
        this.log('TOOL_USED', {
            type: 'tool_usage',
            agentName,
            toolName,
            parameters: toolParams,
            result: this.truncateIfNeeded(result),
            timestamp: new Date().toISOString(),
            ...metadata
        });
    }

    /**
     * Log model usage
     */
    logModelUsage(agentName, modelName, modelType, usage = {}, metadata = {}) {
        if (!this.config.enableModels) return;
        
        this.log('MODEL_USED', {
            type: 'model_usage',
            agentName,
            modelName,
            modelType,
            usage, // tokens, cost, etc.
            timestamp: new Date().toISOString(),
            ...metadata
        });
    }

    /**
     * Log prompts
     */
    logPrompt(agentName, promptType, promptContent, variables = {}, metadata = {}) {
        if (!this.config.enablePrompts) return;
        
        this.log('PROMPT_USED', {
            type: 'prompt',
            agentName,
            promptType,
            promptContent: this.truncateIfNeeded(promptContent),
            variables,
            timestamp: new Date().toISOString(),
            ...metadata
        });
    }

    /**
     * Core logging method
     */
    log(category, data) {
        const handler = this.logHandlers[this.config.outputFormat];
        if (handler) {
            handler(category, data);
        }
    }

    /**
     * Console logging handler with colors
     */
    logToConsole(category, data) {
        const color = this.getCategoryColor(category);
        const resetColor = '\x1b[0m';
        
        // Format the log entry
        const timestamp = new Date().toISOString();
        const prefix = `${color}[${timestamp}] [${category}]${resetColor}`;
        
        // Pretty print the data
        const formattedData = util.inspect(data, {
            depth: null,
            colors: true,
            compact: false
        });
        
        console.log(`${prefix}\n${formattedData}\n`);
    }

    /**
     * File logging handler
     */
    logToFile(category, data) {
        const logEntry = {
            timestamp: new Date().toISOString(),
            category,
            data
        };
        
        const logLine = JSON.stringify(logEntry) + '\n';
        
        try {
            // Check file size and rotate if needed
            if (fs.existsSync(this.config.logFilePath)) {
                const stats = fs.statSync(this.config.logFilePath);
                if (stats.size >= this.config.maxFileSize) {
                    this.rotateLogFile();
                }
            }
            
            fs.appendFileSync(this.config.logFilePath, logLine, 'utf8');
        } catch (error) {
            console.error('Failed to write to log file:', error);
        }
    }

    /**
     * Log to both console and file
     */
    logToBoth(category, data) {
        this.logToConsole(category, data);
        this.logToFile(category, data);
    }

    /**
     * Rotate log file when it gets too large
     */
    rotateLogFile() {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const rotatedPath = this.config.logFilePath.replace('.log', `-${timestamp}.log`);
        
        try {
            fs.renameSync(this.config.logFilePath, rotatedPath);
        } catch (error) {
            console.error('Failed to rotate log file:', error);
        }
    }

    /**
     * Truncate long strings to prevent huge logs
     */
    truncateIfNeeded(value, maxLength = 1000) {
        if (typeof value === 'string' && value.length > maxLength) {
            return value.substring(0, maxLength) + '... [truncated]';
        }
        if (typeof value === 'object' && value !== null) {
            const stringified = JSON.stringify(value);
            if (stringified.length > maxLength) {
                return stringified.substring(0, maxLength) + '... [truncated]';
            }
        }
        return value;
    }

    /**
     * Get color codes for different categories
     */
    getCategoryColor(category) {
        const colors = {
            'ACTIVE_AGENT': '\x1b[36m',   // Cyan
            'AGENT_INPUT': '\x1b[32m',    // Green
            'AGENT_OUTPUT': '\x1b[33m',   // Yellow
            'DECISION': '\x1b[35m',       // Magenta
            'TOOL_USED': '\x1b[34m',      // Blue
            'MODEL_USED': '\x1b[31m',     // Red
            'PROMPT_USED': '\x1b[37m'     // White
        };
        return colors[category] || '\x1b[0m';
    }

    /**
     * Enable/disable specific log categories
     */
    setLogCategory(category, enabled) {
        const categoryMap = {
            'activeAgent': 'enableActiveAgent',
            'agentIO': 'enableAgentIO',
            'decisions': 'enableDecisions',
            'tools': 'enableTools',
            'models': 'enableModels',
            'prompts': 'enablePrompts'
        };
        
        if (categoryMap[category]) {
            this.config[categoryMap[category]] = enabled;
        }
    }

    /**
     * Get current configuration
     */
    getConfig() {
        return { ...this.config };
    }

    /**
     * Parse log file (utility method)
     */
    static async parseLogFile(logFilePath) {
        try {
            const content = fs.readFileSync(logFilePath, 'utf8');
            const lines = content.trim().split('\n');
            return lines.map(line => {
                try {
                    return JSON.parse(line);
                } catch (e) {
                    return null;
                }
            }).filter(Boolean);
        } catch (error) {
            console.error('Failed to parse log file:', error);
            return [];
        }
    }
}

module.exports = AgentLogger;