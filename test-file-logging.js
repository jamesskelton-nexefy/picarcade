/**
 * Test file logging capability of AgentLogger
 */

const AgentLogger = require('./agentLogger.node.js');
const fs = require('fs');

// Initialize logger with file output
const logger = new AgentLogger({
    outputFormat: 'both', // Log to both console and file
    logFilePath: './logs/agent-activity.log'
});

// Create a test agent
const testAgent = {
    name: 'FileTestAgent',
    id: 'agent-file-001'
};

console.log('Testing file logging capability...\n');

// Log various activities
logger.logActiveAgent(testAgent.name, testAgent.id, {
    environment: 'production',
    version: '1.0.0'
});

logger.logAgentInput(testAgent.name, 'Process customer order #12345', {
    source: 'API',
    priority: 'high'
});

logger.logDecision(testAgent.name, 'validate_order', 
    'Order validation required before processing', {
    riskScore: 0.2
});

logger.logToolUsage(testAgent.name, 'database_query', 
    { table: 'orders', id: '12345' },
    { status: 'found', data: { total: 99.99, items: 3 } }
);

logger.logModelUsage(testAgent.name, 'gpt-3.5-turbo', 'language_model', {
    inputTokens: 85,
    outputTokens: 120,
    cost: 0.0015
});

logger.logPrompt(testAgent.name, 'order_confirmation', 
    'Generate a friendly order confirmation message for order {{orderId}} with total {{total}}',
    { orderId: '12345', total: '$99.99' }
);

logger.logAgentOutput(testAgent.name, 
    'Order #12345 has been successfully processed. Total: $99.99',
    { deliveryTime: '2-3 business days' }
);

// Check if log file was created
setTimeout(() => {
    if (fs.existsSync('./logs/agent-activity.log')) {
        console.log('\n✅ Log file created successfully!');
        console.log('📁 Log file location: ./logs/agent-activity.log');
        
        // Display file contents
        const logContent = fs.readFileSync('./logs/agent-activity.log', 'utf8');
        const lines = logContent.trim().split('\n');
        console.log(`\n📊 Total log entries: ${lines.length}`);
        console.log('\n--- Sample log entry (JSON format) ---');
        console.log(lines[0]);
    }
}, 100);