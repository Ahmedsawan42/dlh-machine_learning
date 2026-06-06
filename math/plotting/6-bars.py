#!/usr/bin/env python3

"""
Bar Graph Plotting Module.
"""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Bar Graph Plotting Function.

    Parameters: None
    Returns: None
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    fruits = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    # Create bottom array to stack the bars
    bottom = np.zeros(3)

    # Plot each fruit as a stacked bar
    for i in range(len(fruits)):
        plt.bar(people, fruit[i], width=0.5, bottom=bottom,
                color=colors[i], label=fruits[i])
        bottom += fruit[i]

    plt.ylabel('Quantity of Fruit')
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)
    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.show()
