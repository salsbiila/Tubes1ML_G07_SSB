import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.widgets import Button
from scipy import stats

class StaticVisualizer:
    @staticmethod
    def visualize_network(model, show_weights=True, show_gradients=True, figsize=(12, 10), enable_zoom=True):
        
        G = nx.DiGraph()

        layer_sizes = model.layer_sizes
        max_neurons = max(layer_sizes)

        # dictionary untuk mapping posisi tiap node 
        pos = {}

        horizontal_spacing = 10

        all_nodes = []
        layers_dict = {}

        for i, layer_size in enumerate(layer_sizes) :
            layer_nodes = []

            # vertical spacing size berdaarkan jumlah neuron di layer
            vertical_spacing = 0

            if (layer_size < 10) :
                vertical_spacing = 1.5
            else :
                vertical_spacing = 10 / layer_size

            vertical_offset = (layer_size - 1) * vertical_spacing / 2

            for j in range(layer_size) :
                if i == 0 :
                    node_id = f"i{j}"
                    node_label = f"Input\n{j+1}"
                elif i == len(layer_sizes) - 1 :
                    node_id = f"o{j}" 
                    node_label = f"Output\n{j+1}"
                else : 
                    node_id = f"h{i-1}_{j}"
                    node_label = f"H{i}\n{j+1}"
                
                G.add_node(node_id, label=node_label, layer=i)
                layer_nodes.append(node_id)
                all_nodes.append(node_id)

                # menghitung posisi di canvas
                pos[node_id] = (i * horizontal_spacing, vertical_offset - j * vertical_spacing)

            layers_dict[i] = layer_nodes

        # buat node bisa
        for i in range(1, len(layer_sizes)):
            bias_id = f"b{i-1}"
            G.add_node(bias_id, label=f"Bias {i}", layer=i-0.5)
            all_nodes.append(bias_id)
            
            source_layer_idx = i - 1 
            
            # memposisikan node bias di atas corresponding layer
            if source_layer_idx in layers_dict and layers_dict[source_layer_idx]:
                first_node = layers_dict[source_layer_idx][0]
                
                pos[bias_id] = (
                    pos[first_node][0],  # posisi x sama dengan neuron di corresponding layer
                    pos[first_node][1] + 1.5 
                )
            else:
                # fallback position
                pos[bias_id] = ((i-1) * horizontal_spacing, vertical_offset + 1.5)

        # menghubungkan layer-layer
        for i in range(1, len(layer_sizes)) :
            prev_layer = i - 1
            curr_layer = i

            for j in range(layer_sizes[prev_layer]) :
                for k in range(layer_sizes[curr_layer]) :
                    if prev_layer == 0:
                        source = f"i{j}"
                    else : 
                        source = f"h{prev_layer - 1}_{j}"

                    if curr_layer == len(layer_sizes) - 1:
                        target = f"o{k}"
                    else:
                        target = f"h{curr_layer-1}_{k}"
                    
                    # ambil nliai bobot
                    weight_val = model.weights[i][j, k]

                    # ambil nilai gradien bobot
                    gradient_val = None
                    if hasattr(model, 'weight_gradients') and i in model.weight_gradients:
                        gradient_val = model.weight_gradients[i][j, k]
                    
                    # tambah edge dengan informasi bobot dan gradien bobot
                    G.add_edge(source, target, weight=weight_val, gradient=gradient_val)
            
            # ambil nilai bias
            bias_id = f"b{i-1}"
            for k in range(layer_sizes[curr_layer]) :
                if curr_layer == len(layer_sizes) - 1:
                    target = f"o{k}"
                else: 
                    target = f"h{curr_layer-1}_{k}"

                bias_val = model.biases[i][0, k]

                # ambil gradien bias
                bias_gradient_val = None
                if hasattr(model, 'bias_gradients') and i in model.bias_gradients:
                    bias_gradient_val = model.bias_gradients[i][0, k]
                
                # tambah edge dengan informasi bobot bias dan gradien bobot bias
                G.add_edge(bias_id, target, weight=bias_val, gradient=bias_gradient_val)
        
        # buat figure
        if enable_zoom:
            fig = plt.figure(figsize=figsize)

            # main axis
            ax = plt.axes([0.1, 0.15, 0.8, 0.75])

            # axes untuk zoom & reset buttons
            zoom_in_ax = plt.axes([0.4, 0.05, 0.1, 0.05])
            zoom_out_ax = plt.axes([0.5, 0.05, 0.1, 0.05])
            reset_ax = plt.axes([0.6, 0.05, 0.1, 0.05])
        else:
            fig = plt.figure(figsize=figsize)
            ax = plt.gca()

        layer_colors = ['#4285F4', '#34A853', '#FBBC05', '#EA4335']
        node_colors = []

        for node in all_nodes : 
            layer = G.nodes[node]['layer']
        
            # cycling warna untuk nodes
            color_idx = int(layer) % len(layer_colors)
            
            # warna bias node
            if isinstance(layer, float) and layer % 1 != 0:
                node_colors.append('#7986CB') 
            else:
                node_colors.append(layer_colors[color_idx])
        
        # draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=all_nodes, node_color=node_colors, 
                            node_size=700, alpha=0.8, ax=ax)
        
        # draw node labels
        nx.draw_networkx_labels(G, pos, labels={node: G.nodes[node]['label'] for node in all_nodes},
                            font_size=8, font_weight='bold', ax=ax)
    
        # draw edges
        for u, v, data in G.edges(data=True):
            weight = data['weight']

            edge_color = 'green'
            edge_width = 2
            
            # draw edge
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=edge_width,
                                edge_color=edge_color, alpha=0.6, ax=ax,
                                arrows=True, arrowsize=15)

        # memberi identifier untuk weight label
        def generate_weight_id(source, target):
            if source.startswith('i'):
                source_type = 'i'
                source_num = int(source[1:]) + 1
            elif source.startswith('h'):
                layer_neuron = source[1:].split('_')
                source_type = 'h' + str(int(layer_neuron[0]) + 1)
                source_num = int(layer_neuron[1]) + 1
            elif source.startswith('b'):
                source_type = 'b'
                source_num = int(source[1:]) + 1
            else:
                source_type = 'o'
                source_num = int(source[1:]) + 1
                

            if target.startswith('h'):
                layer_neuron = target[1:].split('_')
                target_type = 'h' + str(int(layer_neuron[0]) + 1)
                target_num = int(layer_neuron[1])+1
            else:  
                target_type = 'o'
                target_num = int(target[1:])+1
                
            return f"W_{source_type}{source_num}_{target_type}{target_num}"

        # edge label untuk weight dan gradient
        if show_weights or show_gradients :
            for u, v, data in G.edges(data=True):
                if u not in pos or v not in pos:
                    continue

                # menentukan posisi
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                
                # menghitung midpoint dengan offset
                dx = x2 - x1
                dy = y2 - y1
                edge_len = np.sqrt(dx*dx + dy*dy)
                
                # normalisasi arah vektor
                dx, dy = dx / edge_len, dy / edge_len
                px, py = -dy, dx

                # implement offset
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2  # midpoint
                label_x = mx + px * 0.4
                label_y = my + py * 0.4

                weight_id = generate_weight_id(u, v)
                
                # text label untuk weight
                label = ""
                if show_weights:
                    label += f"{weight_id}: {data['weight']:.2f}"
                
                if show_gradients and 'gradient' in data and data['gradient'] is not None:
                    if label:
                        label += "\n"
                    label += f"∇{weight_id}: {data['gradient']:.2f}"
                
                if label:
                    ax.text(label_x, label_y, label, 
                           fontsize=4,
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round'),
                           ha='center', va='center', clip_on=True)
        
        # plot title
        ax.set_title(f"Neural Network Architecture\nJumlah Neuron Tiap Layer : {' → '.join([str(size) for size in layer_sizes])}")
        plt.axis('off')

        # informasi fungsi aktivasi pilihan di tiap layer
        if hasattr(model, 'activation') and isinstance(model.activation, list):
            activations_text = "Fungsi Aktivasi:\n"
            for i, act in enumerate(model.activation):
                layer_name = "Hidden" if i < len(model.activation) - 1 else "Output"
                activations_text += f"{layer_name} L. {i+1}: {act}\n"
            
            plt.figtext(0.01, 0.01, activations_text, wrap=True, fontsize=8)
        
        # initial limit sebelum zooming
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # zoom functionality
        if enable_zoom:
            # zoom & reset button
            zoom_in_button = Button(zoom_in_ax, 'Zoom In')
            zoom_out_button = Button(zoom_out_ax, 'Zoom Out')
            reset_button = Button(reset_ax, 'Reset')
        
            def zoom_in(event):
                # ambil nilai limit saat ini
                curr_xlim = ax.get_xlim()
                curr_ylim = ax.get_ylim()
                
                new_xlim = [curr_xlim[0] * 0.8, curr_xlim[1] * 0.8]
                new_ylim = [curr_ylim[0] * 0.8, curr_ylim[1] * 0.8]
                
                # set new limit
                ax.set_xlim(new_xlim)
                ax.set_ylim(new_ylim)
                
                fig.canvas.draw_idle()
            
            def zoom_out(event):
                curr_xlim = ax.get_xlim()
                curr_ylim = ax.get_ylim()
                
                new_xlim = [curr_xlim[0] * 1.2, curr_xlim[1] * 1.2]
                new_ylim = [curr_ylim[0] * 1.2, curr_ylim[1] * 1.2]
                
                ax.set_xlim(new_xlim)
                ax.set_ylim(new_ylim)
                
                fig.canvas.draw_idle()
            
            def reset_view(event):
                # mengembalikan original limit
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                
                fig.canvas.draw_idle()
            
            # Connect the functions to the buttons
            zoom_in_button.on_clicked(zoom_in)
            zoom_out_button.on_clicked(zoom_out)
            reset_button.on_clicked(reset_view)
            
            # store the buttons to prevent garbage collection
            fig.zoom_in_button = zoom_in_button
            fig.zoom_out_button = zoom_out_button
            fig.reset_button = reset_button

        if enable_zoom:
            fig.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
        else:
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        plt.show()
        
        return fig
    
    @staticmethod
    def plot_weight_distribution(model, layers=None, include_bias=True):
        if layers is None:
            # kalau tidak ada input layer, display distribusi semua layer
            layers = list(range(1, len(model.layer_sizes)))
        
        valid_layers = []
        for layer in layers: 
            if layer >= 1 and layer < len(model.layer_sizes):
                valid_layers.append(layer)
            else:
                print(f"Layer {layer} is out of range and will be skipped.")
        
        if not valid_layers:
            print("No valid layers to plot.")
            return None

        # set up the figure & axes
        n_layers = len(valid_layers)
        fig_cols = min(3, n_layers)  # 3 columns di canvas
        fig_rows = (n_layers + fig_cols - 1) // fig_cols  
        
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(5*fig_cols, 4*fig_rows))
        fig.suptitle('Weight Distributions by Layer', fontsize=16)

        if n_layers == 1:
            axes_flat = [axes]
        elif n_layers > 1:
            if fig_rows == 1 or fig_cols == 1:
                if not isinstance(axes, np.ndarray):
                    axes_flat = [axes]
                else: 
                    axes_flat = axes.flatten()
            else:
                axes_flat = axes.flatten()

        for i, layer in enumerate(valid_layers):
            if i < len(axes_flat):
                ax = axes_flat[i]
               
                weights = model.weights[layer].flatten()
                
                if include_bias:
                    biases = model.biases[layer].flatten()
                    all_params = np.concatenate([weights, biases])
                    
                    # plot histogram
                    ax.hist(all_params, bins=30, alpha=0.7, color='blue', 
                            label='Weights (incl. Bias)', density=True)
                    
                    # KDE line untuk distribution shape
                    kde = stats.gaussian_kde(all_params)
                    x_grid = np.linspace(min(all_params), max(all_params), 1000)
                    ax.plot(x_grid, kde(x_grid), 'r-', linewidth=2, label='Density')
                    
                    # calculate stats based on all parameters
                    data_for_stats = all_params
                else:
                    # plot histogram with density=True
                    ax.hist(weights, bins=30, alpha=0.7, color='blue', density=True)
                    
                    kde = stats.gaussian_kde(weights)
                    x_grid = np.linspace(min(weights), max(weights), 1000)
                    ax.plot(x_grid, kde(x_grid), 'r-', linewidth=2, label='Density')
                    
                    # calculate stats based on weights only
                    data_for_stats = weights
                
                # statistics
                mean = np.mean(data_for_stats)
                std = np.std(data_for_stats)
                min_val = np.min(data_for_stats)
                max_val = np.max(data_for_stats)
                # skewness to determine distribution shape
                skewness = stats.skew(data_for_stats)
                
                if abs(skewness) < 0.5:
                    shape = "Normal"
                elif skewness > 0.5:
                    shape = "Right-skewed"
                else:
                    shape = "Left-skewed"
                
                stats_text = (f'Mean: {mean:.4f}\nStd Dev: {std:.4f}\n'
                            f'Min: {min_val:.4f}\nMax: {max_val:.4f}\n'
                            f'Skewness: {skewness:.4f}\nShape: {shape}')
                
                ax.text(0.05, 0.95, stats_text, 
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # garis MEAN
                ax.axvline(mean, color='green', linestyle='dashed', linewidth=1, label='Mean')
                
                ax.set_title(f'Layer {layer} Weight Distribution')
                ax.set_xlabel('Weight Value')
                ax.set_ylabel('Density')
                ax.legend(loc='upper right')
        
        if n_layers > 1:
            for j in range(i+1, len(axes_flat)):
                axes_flat[j].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])  
        plt.show()
        
        return fig
    
    @staticmethod
    def plot_gradient_weight_distribution(model, layers=None, include_bias=True):
        if layers is None:
            layers = list(range(1, len(model.layer_sizes)))
        
        valid_layers = [layer for layer in layers if 1 <= layer < len(model.layer_sizes)]
        if not valid_layers:
            print("No valid layers to plot.")
            return None
        
        n_layers = len(valid_layers)
        fig_cols = min(3, n_layers)
        fig_rows = (n_layers + fig_cols - 1) // fig_cols
        
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(5*fig_cols, 4*fig_rows))
        fig.suptitle('Gradient Weight Distributions by Layer', fontsize=16)
        
        axes_flat = [axes] if n_layers == 1 else axes.flatten()
        
        for i, layer in enumerate(valid_layers):
            if i < len(axes_flat):
                ax = axes_flat[i]
                
                gradients = model.weight_gradients[layer].flatten()
                
                if include_bias:
                    bias_grads = model.bias_gradients[layer].flatten()
                    all_params = np.concatenate([gradients, bias_grads])
                else:
                    all_params = gradients
                
                ax.hist(all_params, bins=30, alpha=0.7, color='blue', density=True, label='Gradients (incl. Bias)')
                
                if len(np.unique(all_params)) > 1:
                    kde = stats.gaussian_kde(all_params)
                    x_grid = np.linspace(min(all_params), max(all_params), 1000)
                    ax.plot(x_grid, kde(x_grid), 'r-', linewidth=2, label='Density')
                
                mean, std = np.mean(all_params), np.std(all_params)
                min_val, max_val = np.min(all_params), np.max(all_params)
                skewness = stats.skew(all_params)
                shape = "Normal" if abs(skewness) < 0.5 else ("Right-skewed" if skewness > 0.5 else "Left-skewed")
                
                stats_text = (f'Mean: {mean:.4f}\nStd Dev: {std:.4f}\n'
                            f'Min: {min_val:.4f}\nMax: {max_val:.4f}\n'
                            f'Skewness: {skewness:.4f}\nShape: {shape}')
                
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax.axvline(mean, color='green', linestyle='dashed', linewidth=1, label='Mean')
                ax.set_title(f'Layer {layer} Gradient Distribution')
                ax.set_xlabel('Gradient Value')
                ax.set_ylabel('Density')
                ax.legend(loc='upper right')
        
        for ax in axes_flat[len(valid_layers):]:
            ax.set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])  
        plt.show()
        
        return fig
