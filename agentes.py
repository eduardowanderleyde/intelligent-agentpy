import agentpy as ap
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class SocialAgent(ap.Agent):
    def setup(self):
        self.opinion = self.model.p.initial_opinion
        self.node = None  # Adiciona o atributo de nó ao agente

    def update_opinion(self):
        neighbors = list(self.model.network.graph.neighbors(self.node))
        neighbor_opinions = [self.model.network.nodes[n]['agent'].opinion for n in neighbors]
        if neighbor_opinions:
            self.opinion = (self.opinion + sum(neighbor_opinions) / len(neighbor_opinions)) / 2

class OpinionModel(ap.Model):
    def setup(self):
        graph = nx.barabasi_albert_graph(self.p.size, self.p.avg_degree)
        self.network = ap.Network(self, graph)
        self.agents = ap.AgentList(self, self.p.size, SocialAgent)
        print(f"Agentes criados: {len(self.agents)}")

        # Associe cada agente a um nó no grafo
        for agent, node in zip(self.agents, self.network.nodes):
            self.network.nodes[node]['agent'] = agent
            agent.node = node
        
        print(f"Nós na rede: {len(self.network.nodes)}")
        initially_influenced = self.agents.random(int(self.p.initial_influenced_share * self.p.size))
        for agent in initially_influenced:
            agent.opinion = 1
        print("Setup completed")

    def step(self):
        for agent in self.agents:
            agent.update_opinion()

    def update(self):
        influenced_count = sum(agent.opinion == 1 for agent in self.agents)
        if influenced_count == 0:
            self.stop()

    def end(self):
        self.data = {'Final Opinions': [agent.opinion for agent in self.agents]}

    def run(self, steps=10000):
        self.setup()
        self.t = 0
        self.running = True
        for step in range(steps):
            self.step()
            self.update()
            if not self.running:
                break
            self.t += 1
        self.end()
        return self.data

parameters = {
    'size': 100,
    'avg_degree': 2,
    'initial_influenced_share': 0.1,
    'initial_opinion': 0
}

model = OpinionModel(parameters)
results = model.run(steps=10000)

if 'Final Opinions' in results:
    nomes = [f'Agente {i}' for i in range(len(results['Final Opinions']))]
    idades = [28, 24, 35, 32] * (len(results['Final Opinions']) // 4 + 1)
    cidades = ['New York', 'Paris', 'Berlin', 'London'] * (len(results['Final Opinions']) // 4 + 1)

    data = {
        'Nome': nomes,
        'Idade': idades[:len(nomes)],
        'Cidade': cidades[:len(nomes)],
        'Opinião Final': results['Final Opinions']
    }
    df = pd.DataFrame(data)
    print(df)
else:
    print("Nenhuma opinião final registrada.")

def plot_opinions(data):
    sns.histplot(data, kde=False, bins=10)
    plt.xlabel('Opinião')
    plt.ylabel('Contagem')
    plt.title('Distribuição das Opiniões')
    plt.show()

plot_opinions(df['Opinião Final'])
