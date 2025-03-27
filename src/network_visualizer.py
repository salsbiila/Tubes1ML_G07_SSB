import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.widgets import Button

class NetworkVisualizer:
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
                    node_id = f"i{j}" # input layer
                    node_label = f"Input\n{j+1}"
                elif i == len(layer_sizes) - 1 :
                    node_id = f"o{j}" # output layer
                    node_label = f"Output\n{j+1}"
                else : 
                    node_id = f"h{i-1}_{j}" # hidden layer
                    node_label = f"H{i}\n{j+1}"
                
                G.add_node(node_id, label=node_label, layer=i)
                layer_nodes.append(node_id)
                all_nodes.append(node_id)

                # menghitung posisi di canvas
                pos[node_id] = (i * horizontal_spacing, vertical_offset - j * vertical_spacing)

            layers_dict[i] = layer_nodes

        # Position bias nodes directly above their corresponding layer
        for i in range(1, len(layer_sizes)):
            bias_id = f"b{i-1}"
            G.add_node(bias_id, label=f"Bias {i}", layer=i-0.5)
            all_nodes.append(bias_id)
            
            # Position bias nodes directly above their corresponding layer
            source_layer_idx = i - 1  # The layer to which this bias connects from
            
            # Get the first node in this layer
            if source_layer_idx in layers_dict and layers_dict[source_layer_idx]:
                first_node = layers_dict[source_layer_idx][0]
                # Position the bias node above this layer with the same x position
                pos[bias_id] = (
                    pos[first_node][0],  # Same x as the source layer
                    pos[first_node][1] + 1.5  # Above the top node in the layer
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
                    
                    # ambil bobot
                    weight_val = model.weights[i][j, k]

                    gradient_val = None
                    if hasattr(model, 'weight_gradient') and i in model.weight_gradient:
                        gradient_val = model.weight_gradient[i][j, k]
                    
                    # Add edge with weight information
                    G.add_edge(source, target, weight=weight_val, gradient=gradient_val)
            
            bias_id = f"b{i-1}"
            for k in range(layer_sizes[curr_layer]) :
                if curr_layer == len(layer_sizes) - 1:
                    target = f"o{k}"
                else: 
                    target = f"h{curr_layer-1}_{k}"

                bias_val = model.biases[i][0, k]

                # Get bias gradient value if available
                bias_gradient_val = None
                if hasattr(model, 'bias_gradient') and i in model.bias_gradient:
                    bias_gradient_val = model.bias_gradient[i][0, k]
                
                # Add edge with bias information
                G.add_edge(bias_id, target, weight=bias_val, gradient=bias_gradient_val)
        
        # Create figure and axes with special layout if zoom is enabled
        if enable_zoom:
            # Create figure with more space at bottom for zoom controls
            fig = plt.figure(figsize=figsize)
            # Main axis for network plot
            ax = plt.axes([0.1, 0.15, 0.8, 0.75])
            # Create axes for zoom buttons
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
        
            # Use modulo for cycling through colors if there are more layers than colors
            color_idx = int(layer) % len(layer_colors)
            
            # Bias nodes get a different color
            if isinstance(layer, float) and layer % 1 != 0:  # Bias nodes have non-integer layers
                node_colors.append('#7986CB')  # Light indigo for bias
            else:
                node_colors.append(layer_colors[color_idx])
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=all_nodes, node_color=node_colors, 
                            node_size=700, alpha=0.8, ax=ax)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, labels={node: G.nodes[node]['label'] for node in all_nodes},
                            font_size=8, font_weight='bold', ax=ax)
    
        # Draw edges
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            
            # Determine edge color based on weight sign
            # edge_color = 'green' if weight > 0 else 'red'
            edge_color = 'green'
            
            # Determine edge width based on weight magnitude (normalized)
            # edge_width = 1 + 3 * min(1, abs(weight))
            edge_width = 2
            
            # Draw edge
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=edge_width,
                                edge_color=edge_color, alpha=0.6, ax=ax,
                                arrows=True, arrowsize=15)

        # Function to generate a descriptive identifier for weights
        def generate_weight_id(source, target):
            # Parse the source node ID
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
                
            # Parse the target node ID
            if target.startswith('h'):
                layer_neuron = target[1:].split('_')
                target_type = 'h' + str(int(layer_neuron[0]) + 1)
                target_num = int(layer_neuron[1])+1
            else:  # must be output
                target_type = 'o'
                target_num = int(target[1:])+1
                
            # Build the identifier
            return f"W_{source_type}{source_num}_{target_type}{target_num}"

        # edge label untuk weight dan gradient
        if show_weights or show_gradients :
            for u, v, data in G.edges(data=True):
                if u not in pos or v not in pos:
                    continue

                # Get positions
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                
                # Calculate midpoint with offset
                dx = x2 - x1
                dy = y2 - y1
                edge_len = np.sqrt(dx*dx + dy*dy)
                
                # Calculate normalized direction vectors
                dx, dy = dx / edge_len, dy / edge_len
                
                # Perpendicular vector for offset
                px, py = -dy, dx

                # Apply offset (adjust the 0.3 value if needed)
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2  # midpoint
                label_x = mx + px * 0.4
                label_y = my + py * 0.4

                # Create descriptive weight identifier
                weight_id = generate_weight_id(u, v)
                
                # Create label text
                label = ""
                if show_weights:
                    label += f"{weight_id}: {data['weight']:.2f}"
                
                if show_gradients and 'gradient' in data and data['gradient'] is not None:
                    if label:
                        label += "\n"
                    label += f"∇{weight_id}: {data['gradient']:.2f}"
                
                if label:
                    # Create text annotation directly
                    ax.text(label_x, label_y, label, 
                           fontsize=4,
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round'),
                           ha='center', va='center', clip_on=True)
            # edge_labels = {}
            
            # for u, v, data in G.edges(data=True) :
            #     label = ""

            #     # tambah info weight
            #     if show_weights: 
            #         label += f"W: {data['weight']:.2f}"

            #     #tambah info gradient
            #     if show_gradients and 'gradient' in data and data['gradient'] is not None:
            #         if label:
            #             label += "\n"
            #         label += f"∇W: {data['gradient']:.2f}"
                
            #     if label:
            #         edge_labels[(u, v)] = label
            # # edge_label_positions = create_edge_label_positions(G, pos)
            # nx.draw_networkx_edge_labels(
            #     G, 
            #     # edge_label_positions,  # Use custom positions instead of pos
            #     edge_labels=edge_labels, 
            #     font_size=7, 
            #     font_color='black', 
            #     bbox=dict(alpha=0.7, boxstyle='round', ec='gray', fc='white'),
            #     rotate=False  # Do not rotate labels to improve readability
            # )
            # # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='black', bbox=dict(alpha=0.7, boxstyle='round', ec='gray', fc='white'))
        
        # # Create legend
        # legend_elements = [
        #     mpatches.Patch(color='#7986CB', label='Bias'),
        #     mpatches.Patch(color=layer_colors[0], label='Input Layer'),
        #     mpatches.Patch(color=layer_colors[1], label='Hidden Layer 1'),
        # ]
        
        # # Add more legend elements if there are more layers
        # for i in range(2, len(layer_sizes) - 1):
        #     color_idx = i % len(layer_colors)
        #     legend_elements.append(mpatches.Patch(color=layer_colors[color_idx], 
        #                                         label=f'Hidden Layer {i}'))
        
        # # Add output layer to legend
        # legend_elements.append(mpatches.Patch(
        #     color=layer_colors[(len(layer_sizes) - 1) % len(layer_colors)], 
        #     label='Output Layer'))
        
        # # Add legend to plot
        # ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Set plot title and remove axis
        ax.set_title(f"Neural Network Architecture\nJumlah Neuron Tiap Layer : {' → '.join([str(size) for size in layer_sizes])}")
        plt.axis('off')

        # Add activation functions information
        if hasattr(model, 'activation') and isinstance(model.activation, list):
            activations_text = "Fungsi Aktivasi:\n"
            for i, act in enumerate(model.activation):
                layer_name = "Hidden" if i < len(model.activation) - 1 else "Output"
                activations_text += f"{layer_name} L. {i+1}: {act}\n"
            
            plt.figtext(0.01, 0.01, activations_text, wrap=True, fontsize=8)
        
        # Get initial limits before zooming
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Add zoom functionality if enabled
        if enable_zoom:
            # Create zoom in button
            zoom_in_button = Button(zoom_in_ax, 'Zoom In')
            zoom_out_button = Button(zoom_out_ax, 'Zoom Out')
            reset_button = Button(reset_ax, 'Reset')
            
            # Define zoom functions
            def zoom_in(event):
                # Get current limits
                curr_xlim = ax.get_xlim()
                curr_ylim = ax.get_ylim()
                
                # Calculate new limits (zoom in by 20%)
                new_xlim = [curr_xlim[0] * 0.8, curr_xlim[1] * 0.8]
                new_ylim = [curr_ylim[0] * 0.8, curr_ylim[1] * 0.8]
                
                # Set new limits
                ax.set_xlim(new_xlim)
                ax.set_ylim(new_ylim)
                
                # Redraw
                fig.canvas.draw_idle()
            
            def zoom_out(event):
                # Get current limits
                curr_xlim = ax.get_xlim()
                curr_ylim = ax.get_ylim()
                
                # Calculate new limits (zoom out by 20%)
                new_xlim = [curr_xlim[0] * 1.2, curr_xlim[1] * 1.2]
                new_ylim = [curr_ylim[0] * 1.2, curr_ylim[1] * 1.2]
                
                # Set new limits
                ax.set_xlim(new_xlim)
                ax.set_ylim(new_ylim)
                
                # Redraw
                fig.canvas.draw_idle()
            
            def reset_view(event):
                # Restore original limits
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                
                # Redraw
                fig.canvas.draw_idle()
            
            # Connect the functions to the buttons
            zoom_in_button.on_clicked(zoom_in)
            zoom_out_button.on_clicked(zoom_out)
            reset_button.on_clicked(reset_view)
            
            # Store the buttons to prevent garbage collection
            fig.zoom_in_button = zoom_in_button
            fig.zoom_out_button = zoom_out_button
            fig.reset_button = reset_button

        if enable_zoom:
            fig.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
        else:
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        plt.show()
        
        return fig
    