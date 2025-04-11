import numpy as np
import networkx as nx
from scipy import stats
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

class InteractiveVisualizer:
    @staticmethod
    def convert(nid):
        if nid.startswith("i"):
            return f"i{int(nid[1:]) + 1}"
        elif nid.startswith("h"):
            parts = nid[1:].split("_")
            return f"h{int(parts[0]) + 1}{int(parts[1]) + 1}"
        elif nid.startswith("o"):
            return f"o{int(nid[1:]) + 1}"
        elif nid.startswith("b"):
            return f"b{int(nid[1:]) + 1}"
        return nid

    @staticmethod
    def get_weight_label(source, target, is_bias=False):
        src = InteractiveVisualizer.convert(source)
        tgt = InteractiveVisualizer.convert(target)
        return f"{'B' if is_bias else 'W'}_{src}_{tgt}"

    @staticmethod
    def visualize_network(model, show_gradients=True):
        G = nx.DiGraph()
        layer_sizes = model.layer_sizes
        num_layers = len(layer_sizes)

        # Dynamic colors for layers
        layer_colors = sample_colorscale("Viridis", [i / max(1, num_layers - 1) for i in range(num_layers)])

        pos = {}
        horizontal_spacing = 200
        vertical_spacing = 200
        node_labels = {}
        node_colors = []
        annotations = []

        # Create neuron nodes
        for i, layer_size in enumerate(layer_sizes):
            vertical_offset = (layer_size - 1) * vertical_spacing / 2
            for j in range(layer_size):
                if i == 0:
                    node_id = f"i{j}"
                    label = f"Input {j+1}"
                elif i == num_layers - 1:
                    node_id = f"o{j}"
                    label = f"Output {j+1}"
                else:
                    node_id = f"h{i-1}_{j}"
                    label = f"H{i} {j+1}"

                x = i * horizontal_spacing
                y = vertical_offset - j * vertical_spacing
                pos[node_id] = (x, y)
                G.add_node(node_id, layer=i)
                node_labels[node_id] = label
                node_colors.append(layer_colors[i])

                # Permanent neuron label annotation
                annotations.append(dict(
                    x=x, y=y,
                    text=label,
                    showarrow=False,
                    font=dict(size=12, color='black'),
                    xanchor='center',
                    yanchor='middle',
                    bgcolor='rgba(255,255,255,0.5)'  # semi-transparent
                ))

        for i in range(1, num_layers):
            bias_id = f"b{i-1}"
            bias_label = f"Bias {i}"

            if i - 1 == 0:
                first_prev = "i0"
            else:
                first_prev = f"h{i - 2}_0"

            x = (i - 1) * horizontal_spacing
            y = pos[first_prev][1] + vertical_spacing * 1.5
            pos[bias_id] = (x, y)
            G.add_node(bias_id, layer=i - 0.5)
            node_labels[bias_id] = bias_label
            node_colors.append('#7986CB')  # fixed color for bias

            annotations.append(dict(
                x=x, y=y,
                text=bias_label,
                showarrow=False,
                font=dict(size=10, color='white'),
                xanchor='center',
                yanchor='middle',
                bgcolor='rgba(0,0,0,0.5)'
            ))

        edge_x, edge_y = [], []
        edge_hover_texts = []

        # Create edges
        for i in range(1, num_layers):
            prev_layer = i - 1
            curr_layer = i

            for j in range(layer_sizes[prev_layer]):
                for k in range(layer_sizes[curr_layer]):
                    source = f"i{j}" if prev_layer == 0 else f"h{prev_layer - 1}_{j}"
                    target = f"o{k}" if curr_layer == num_layers - 1 else f"h{curr_layer - 1}_{k}"

                    weight_val = model.weights[i][j, k]
                    gradient_val = model.weight_gradients[i][j, k] if hasattr(model, 'weight_gradients') else None

                    G.add_edge(source, target)

                    x0, y0 = pos[source]
                    x1, y1 = pos[target]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

                    label = InteractiveVisualizer.get_weight_label(source, target)
                    hover_text = f"{label}: {weight_val:.4f}"
                    if show_gradients and gradient_val is not None:
                        hover_text += f"<br>∇{label}: {gradient_val:.4f}"

                    edge_hover_texts += [hover_text, hover_text, None]

            bias_id = f"b{i-1}"
            for k in range(layer_sizes[curr_layer]):
                target = f"o{k}" if curr_layer == num_layers - 1 else f"h{curr_layer - 1}_{k}"
                bias_val = model.biases[i][0, k]
                bias_grad = model.bias_gradients[i][0, k] if hasattr(model, 'bias_gradients') else None

                G.add_edge(bias_id, target)

                x0, y0 = pos[bias_id]
                x1, y1 = pos[target]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

                label = InteractiveVisualizer.get_weight_label(bias_id, target, is_bias=True)
                hover_text = f"{label}: {bias_val:.4f}"
                if show_gradients and bias_grad is not None:
                    hover_text += f"<br>∇{label}: {bias_grad:.4f}"

                edge_hover_texts += [hover_text, hover_text, None]

        # Plot edges
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='gray'),
            hoverinfo='text',
            mode='lines',
            text=edge_hover_texts
        )

        # Invisible hover points
        hover_nodes_x = []
        hover_nodes_y = []
        hover_texts = []

        for i in range(0, len(edge_x), 3):
            if edge_x[i] is not None and edge_x[i+1] is not None:
                mx = (edge_x[i] + edge_x[i+1]) / 2
                my = (edge_y[i] + edge_y[i+1]) / 2
                hover_nodes_x.append(mx)
                hover_nodes_y.append(my)
                hover_texts.append(edge_hover_texts[i])

        hover_trace = go.Scatter(
            x=hover_nodes_x,
            y=hover_nodes_y,
            mode='markers',
            text=hover_texts,
            hoverinfo='text',
            marker=dict(size=20, color='rgba(0,0,0,0)', line=dict(width=1, color='rgba(0,0,0,0)')),
            showlegend=False
        )

        # Plot nodes
        node_x, node_y = zip(*[pos[n] for n in G.nodes()])
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='none',
            marker=dict(
                color=node_colors,
                size=50,
                line=dict(width=1, color='black')
            )
        )

        # Create subtitle
        layer_info = " → ".join(str(size) for size in model.layer_sizes)
        title_text = f"<b>Neural Network Architecture</b><br><span style='font-size:14px'>Neurons per Layer: {layer_info}</span>"

        fig_height = max(800, vertical_spacing * max(layer_sizes))
        fig_width = max(1400, (len(layer_sizes) - 1) * horizontal_spacing)

        # Final figure
        fig = go.Figure(data=[edge_trace, hover_trace, node_trace],
                        layout=go.Layout(
                            title={
                                "text": title_text,
                                "x": 0.5,
                                "xanchor": "center",
                                "y": 0.95,
                                "yanchor": "top"
                            },
                            height= fig_height,
                            width= fig_width,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=20, r=20, t=100),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            plot_bgcolor='whitesmoke',
                            annotations=annotations
                        ))
        fig.show()
        return fig

    @staticmethod
    def plot_weight_distribution(model, layers=None, include_bias=True):
        if layers is None:
            # kalau tidak ada input layer, display distribusi semua layer
            layers = list(range(1, len(model.layer_sizes)))

        valid_layers = []
        for layer in layers:
            if 1 <= layer < len(model.layer_sizes):
                valid_layers.append(layer)
            else:
                print(f"Layer {layer} is out of range and will be skipped.")

        if not valid_layers:
            print("No valid layers to plot.")
            return None

        traces = []
        buttons = []
        annotations = []

        for i, layer in enumerate(valid_layers):
            weights = model.weights[layer].flatten()

            if include_bias:
                biases = model.biases[layer].flatten()
                all_params = np.concatenate([weights, biases])
                data_for_stats = all_params
                hist_label = "Weights (incl. Bias)"
            else:
                data_for_stats = weights
                hist_label = "Weights only"

            # Histogram
            hist = go.Histogram(
                x=data_for_stats,
                nbinsx=30,
                name=hist_label,
                histnorm='probability density',
                opacity=0.7,
                marker=dict(color='blue'),
                visible=(i == 0),
            )

            # KDE Line
            kde = stats.gaussian_kde(data_for_stats)
            x_grid = np.linspace(min(data_for_stats), max(data_for_stats), 1000)
            y_kde = kde(x_grid)

            kde_line = go.Scatter(
                x=x_grid,
                y=y_kde,
                mode='lines',
                name='Density',
                line=dict(color='red'),
                visible=(i == 0)
            )

            # Mean Line as Scatter (so it shows in legend)
            mean = np.mean(data_for_stats)
            mean_line = go.Scatter(
                x=[mean, mean],
                y=[0, max(y_kde) * 1.05],
                mode='lines',
                name='Mean',
                line=dict(color='green', dash='dash'),
                visible=(i == 0)
            )

            # Stats
            std = np.std(data_for_stats)
            min_val = np.min(data_for_stats)
            max_val = np.max(data_for_stats)
            skewness = stats.skew(data_for_stats)

            if abs(skewness) < 0.5:
                shape = "Normal"
            elif skewness > 0.5:
                shape = "Right-skewed"
            else:
                shape = "Left-skewed"

            stats_text = (f'Mean: {mean:.4f}<br>Std Dev: {std:.4f}<br>'
                        f'Min: {min_val:.4f}<br>Max: {max_val:.4f}<br>'
                        f'Skewness: {skewness:.4f}<br>Shape: {shape}')

            annotation = dict(
                text=stats_text,
                x=0.05,
                y=0.95,
                xref='paper',
                yref='paper',
                showarrow=False,
                align='left',
                bgcolor='white',
                bordercolor='black',
                borderwidth=1
            )

            annotations.append(annotation)
            traces.extend([hist, kde_line, mean_line])

            # Make visibility mask for all traces
            visibility_mask = [j // 3 == i for j in range(3 * len(valid_layers))]

            # Button for dropdown
            buttons.append(dict(
                label=f"Layer {layer}",
                method="update",
                args=[
                    {"visible": visibility_mask},
                    {"annotations": [annotation]}
                ]
            ))

        # Create figure
        fig = go.Figure(data=traces)

        # Layout settings
        fig.update_layout(
            title={
                "text": "Weight Distributions by Layer",
                "x": 0.5,
                "xanchor": "center"
            },
            xaxis_title="Weight Value",
            yaxis_title="Density",
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                direction="down",
                x=0.0,
                y=1.15,
                showactive=True
            )],
            annotations=[annotations[0]],
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="center",
                x=0.5
            )
        )

        fig.show()
        return fig
    
    @staticmethod
    def plot_gradient_weight_distribution(model, layers=None, include_bias=True):
        if layers is None:
            layers = list(range(1, len(model.layer_sizes)))

        valid_layers = [layer for layer in layers if 1 <= layer < len(model.layer_sizes)]
        if not valid_layers:
            print("No valid layers to plot.")
            return None

        traces = []
        buttons = []
        annotations = []

        for i, layer in enumerate(valid_layers):
            gradients = model.weight_gradients[layer].flatten()

            if include_bias:
                bias_grads = model.bias_gradients[layer].flatten()
                all_params = np.concatenate([gradients, bias_grads])
                hist_label = "Gradients (incl. Bias)"
            else:
                all_params = gradients
                hist_label = "Gradients only"

            # Histogram
            hist = go.Histogram(
                x=all_params,
                nbinsx=30,
                name=hist_label,
                histnorm='probability density',
                opacity=0.7,
                marker=dict(color='blue'),
                visible=(i == 0),
            )

            # KDE Line (only if the data has some variation)
            if len(np.unique(all_params)) > 1:
                kde = stats.gaussian_kde(all_params)
                x_grid = np.linspace(min(all_params), max(all_params), 1000)
                y_kde = kde(x_grid)

                kde_line = go.Scatter(
                    x=x_grid,
                    y=y_kde,
                    mode='lines',
                    name='Density',
                    line=dict(color='red'),
                    visible=(i == 0)
                )

                # Mean Line (as Scatter so it appears in legend)
                mean = np.mean(all_params)
                mean_line = go.Scatter(
                    x=[mean, mean],
                    y=[0, max(y_kde) * 1.05],
                    mode='lines',
                    name='Mean',
                    line=dict(color='green', dash='dash'),
                    visible=(i == 0)
                )
            else:
                kde_line = go.Scatter(x=[], y=[], visible=False)
                mean_line = go.Scatter(x=[], y=[], visible=False)

            mean = np.mean(all_params)
            std = np.std(all_params)
            min_val = np.min(all_params)
            max_val = np.max(all_params)
            skewness = stats.skew(all_params)

            if abs(skewness) < 0.5:
                shape = "Normal"
            elif skewness > 0.5:
                shape = "Right-skewed"
            else:
                shape = "Left-skewed"

            stats_text = (f'Mean: {mean:.4f}<br>Std Dev: {std:.4f}<br>'
                        f'Min: {min_val:.4f}<br>Max: {max_val:.4f}<br>'
                        f'Skewness: {skewness:.4f}<br>Shape: {shape}')

            annotation = dict(
                text=stats_text,
                x=0.05,
                y=0.95,
                xref='paper',
                yref='paper',
                showarrow=False,
                align='left',
                bgcolor='white',
                bordercolor='black',
                borderwidth=1
            )

            annotations.append(annotation)
            traces.extend([hist, kde_line, mean_line])

            visibility_mask = [j // 3 == i for j in range(3 * len(valid_layers))]

            buttons.append(dict(
                label=f"Layer {layer}",
                method="update",
                args=[
                    {"visible": visibility_mask},
                    {"annotations": [annotation]}
                ]
            ))

        fig = go.Figure(data=traces)

        fig.update_layout(
            title={
                "text": "Gradient Weight Distributions by Layer",
                "x": 0.5,
                "xanchor": "center"
            },
            xaxis_title="Gradient Value",
            yaxis_title="Density",
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                direction="down",
                x=0.0,
                y=1.15,
                showactive=True
            )],
            annotations=[annotations[0]],
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="center",
                x=0.5
            )
        )

        fig.show()
        return fig
    
    def plot_loss_curves(history):
        fig = go.Figure()

        epochs = list(range(1, len(history['train_loss']) + 1))
        # plot training loss
        fig.add_trace(go.Scatter(
            x = epochs,
            y = history['train_loss'],
            mode = 'lines',
            name = f"Training Loss",
            line = dict(color='blue')
        ))

        # plot validation loss
        if 'val_loss' in history and history['val_loss']:
            fig.add_trace(go.Scatter(
                x=epochs,
                y=history['val_loss'],
                mode='lines',
                name=f"Validation Loss",
                line=dict(color='red') 
            ))

        fig.update_layout(
            title={
                "text": "Training Loss",
                "x": 0.5,
                "xanchor": "center"
            },
            xaxis=dict(
                title="Epoch",
                tickfont=dict(size=14)  
            ),
            yaxis=dict(
                title="Loss",
                tickfont=dict(size=14)
            ),
            legend_title="Models",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig