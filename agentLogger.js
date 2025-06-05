/**
 * Agent Logger Module
 * Configurable logging system for agent-based applications
 * Only logs: Active Agent, Agent I/O, Decisions, Tools, Models, and Prompts
 */

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
            outputFormat: config.outputFormat || 'console', // console, file, or custom
            ...config
        };
        
        this.logHandlers = {
            console: this.logToConsole.bind(this),
            file: this.logToFile.bind(this),
            custom: config.customHandler || this.logToConsole.bind(this)
        };
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
            input,
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
            output,
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
            reasoning,
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
            result,
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
            promptContent,
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
     * Console logging handler
     */
    logToConsole(category, data) {
        const color = this.getCategoryColor(category);
        console.log(`${color}[${category}]${this.resetColor()}`, JSON.stringify(data, null, 2));
    }

    /**
     * File logging handler (placeholder - implement based on environment)
     */
    logToFile(category, data) {
        // In a Node.js environment, you would use fs.appendFileSync
        // In a browser environment, you might use localStorage or send to a server
        console.warn('File logging not implemented in browser environment');
        this.logToConsole(category, data);
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

    resetColor() {
        return '\x1b[0m';
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
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AgentLogger;
}