import jsonschema
from jsonschema import validate

simple_perceptron = {
    "type": "object",
    "properties": {
        "epochs": {
            "type": "integer",
            "minimum": 1,
        },
        "learning_rate": {
            "type": "float",
            "minimum": 0,
            "maximum": 1
        },
        "plot": {
            "type": "boolean"
        },
        "stop_early": {
            "type": "boolean"
        },
    },
    "required": ["epochs", "learning_rate", "plot_iteration", "stop_early"]
}

non_linear_perceptron = {
    "type": "object",
    "properties": {
        "epochs": {
            "type": "integer",
            "minimum": 1,
        },
        "learning_rate": {
            "type": "float",
            "minimum": 0,
            "maximum": 1
        },
        "beta": {
            "type": "float",
            "minimum": 0,
            "maximum": 1
        },
        "selection_method": {
            "type": "integer",
            "minimum": 0,
            "maximum": 1
        },
        "stop_early": {
            "type": "boolean"
        },
    },
    "required": ["epochs", "learning_rate", "beta", "selection_method", "stop_early"]
}

multi_layer_perceptron_xor = {
    "type": "object",
    "properties": {
        "epochs": {
            "type": "integer",
            "minimum": 1,
        },
        "layers": {
            "type": "array"
        },
        "plot": {
            "type": "boolean"
        }
    },
    "required": ["epochs", "layers", "plot"]
}

multi_layer_perceptron_primes = {
    "type": "object",
    "properties": {
        "epochs": {
            "type": "integer",
            "minimum": 1,
        },
        "layers": {
            "type": "array"
        },
        "plot": {
            "type": "boolean"
        }
    },
    "required": ["epochs", "layers", "plot"]
}

multi_layer_perceptron_dataset_ex_2 = {
    "type": "object",
    "properties": {
        "epochs": {
            "type": "integer",
            "minimum": 1,
        },
        "layers": {
            "type": "array"
        },
        "plot": {
            "type": "boolean"
        }
    },
    "required": ["epochs", "layers", "plot"]
}

configSchema = {
    "type": "object",
    "properties": {
        "simple_perceptron_and": simple_perceptron,
        "simple_perceptron_xor": simple_perceptron,
        "non_linear_perceptron": non_linear_perceptron,
        "multi_layer_perceptron_xor": multi_layer_perceptron_xor,
        "multi_layer_perceptron_primes":multi_layer_perceptron_primes,
        "multi_layer_perceptron_dataset_ex_2": multi_layer_perceptron_dataset_ex_2
    },
    "required": ["simple_perceptron_and", "simple_perceptron_xor", "non_linear_perceptron", "multi_layer_perceptron_xor", "multi_layer_perceptron_primes", "multi_layer_perceptron_dataset_ex_2"]
}
