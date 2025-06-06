/**
 * Example usage of AgentLogger in an agent-based system
 * Demonstrates logging of only the required information
 */

// Import the AgentLogger
const AgentLogger = require('./agentLogger.node.js');

// Initialize the logger with default settings (all categories enabled)
const logger = new AgentLogger({
    outputFormat: 'console',
    logLevel: 'info'
});

// Example Agent class
class Agent {
    constructor(name, id) {
        this.name = name;
        this.id = id;
        this.logger = logger;
    }

    activate() {
        // Log when agent becomes active
        this.logger.logActiveAgent(this.name, this.id, {
            status: 'activated',
            capabilities: ['text-generation', 'tool-usage']
        });
    }

    processInput(input) {
        // Log agent input
        this.logger.logAgentInput(this.name, input, {
            inputType: 'user_query',
            length: input.length
        });

        // Simulate decision making
        const decision = this.makeDecision(input);
        
        // Process and generate output
        const output = this.generateOutput(decision);
        
        // Log agent output
        this.logger.logAgentOutput(this.name, output, {
            outputType: 'response',
            processingTime: '150ms'
        });

        return output;
    }

    makeDecision(input) {
        const decision = {
            action: 'generate_response',
            useTools: input.includes('search') || input.includes('calculate'),
            confidence: 0.85
        };

        // Log the decision
        this.logger.logDecision(
            this.name,
            decision.action,
            `Based on input analysis, decided to ${decision.action} with ${decision.confidence} confidence`,
            { confidence: decision.confidence, useTools: decision.useTools }
        );

        return decision;
    }

    useTool(toolName, params) {
        // Log tool usage
        this.logger.logToolUsage(
            this.name,
            toolName,
            params,
            { success: true, data: 'tool_result_data' },
            { duration: '200ms' }
        );
    }

    useModel(modelName, promptContent) {
        // Log model usage
        this.logger.logModelUsage(
            this.name,
            modelName,
            'language_model',
            {
                inputTokens: 150,
                outputTokens: 200,
                cost: 0.002
            },
            { provider: 'OpenAI' }
        );

        // Log the prompt used
        this.logger.logPrompt(
            this.name,
            'generation',
            promptContent,
            {
                temperature: 0.7,
                maxTokens: 500
            },
            { templateVersion: 'v2.1' }
        );
    }

    generateOutput(decision) {
        if (decision.useTools) {
            this.useTool('web_search', { query: 'example search' });
        }

        const prompt = "Generate a helpful response based on the user's input";
        this.useModel('gpt-4', prompt);

        return "This is the agent's generated response.";
    }
}

// Example usage
console.log('=== Agent Logger Example ===\n');

// Create and activate an agent
const agent = new Agent('Assistant', 'agent-001');
agent.activate();

// Process some input
console.log('\n--- Processing User Input ---');
agent.processInput('Can you search for information about AI?');

// Disable some logging categories
console.log('\n--- Disabling Some Log Categories ---');
logger.setLogCategory('models', false);
logger.setLogCategory('prompts', false);

// Process another input (models and prompts won't be logged)
console.log('\n--- Processing Another Input (without model/prompt logs) ---');
agent.processInput('Calculate the sum of 5 and 10');

// Show current configuration
console.log('\n--- Current Logger Configuration ---');
console.log(logger.getConfig());