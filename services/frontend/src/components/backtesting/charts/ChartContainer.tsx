/**
 * Advanced Chart Container using TradingView Lightweight Charts
 * Based on the React Advanced example pattern
 */

import React, { 
    createContext, 
    forwardRef, 
    useCallback, 
    useEffect, 
    useImperativeHandle, 
    useLayoutEffect, 
    useRef, 
    useState 
} from 'react';
import { 
    createChart, 
    IChartApi, 
    ISeriesApi, 
    ColorType,
    SeriesType 
} from 'lightweight-charts';
import { ChartTheme, CHART_CONSTANTS } from './types';

// Chart context type
interface ChartContextType {
    _api?: IChartApi;
    isRemoved: boolean;
    paneCount: number;
    api(): IChartApi;
    free(series: ISeriesApi<SeriesType>): void;
    addPane(): number;
    setPaneHeight(paneIndex: number, height: number): void;
    getTotalHeight(): number;
}

// Context for sharing chart API between components
const ChartContext = createContext<ChartContextType | null>(null);

export interface ChartContainerProps {
    children?: React.ReactNode;
    container: HTMLElement;
    theme: ChartTheme;
    height?: number;
    onChartReady?: (chart: IChartApi) => void;
}

export interface ChartProps {
    children?: React.ReactNode;
    theme: ChartTheme;
    height?: number;
    className?: string;
    onChartReady?: (chart: IChartApi) => void;
}

// Main Chart component
export function Chart(props: ChartProps) {
    const [container, setContainer] = useState<HTMLElement | null>(null);
    const handleRef = useCallback((ref: HTMLElement | null) => setContainer(ref), []);
    
    return (
        <div ref={handleRef} className={props.className}>
            {container && <ChartContainer {...props} container={container} />}
        </div>
    );
}

// Chart Container component with proper lifecycle management
export const ChartContainer = forwardRef<IChartApi, ChartContainerProps>((props, ref) => {
    const { children, container, theme, height, onChartReady } = props;

    const chartApiRef = useRef<{
        _api?: IChartApi;
        isRemoved: boolean;
        paneCount: number;
        api(): IChartApi;
        free(series: ISeriesApi<SeriesType>): void;
        addPane(): number;
        setPaneHeight(paneIndex: number, height: number): void;
        getTotalHeight(): number;
    }>({
        isRemoved: false,
        paneCount: 1, // Start with main pane
        
        api() {
            if (!this._api) {
                // Calculate initial height
                const initialHeight = height || CHART_CONSTANTS.DEFAULT_MAIN_PANE_HEIGHT;
                
                this._api = createChart(container, {
                    width: container.clientWidth,
                    height: initialHeight,
                    layout: {
                        background: { type: ColorType.Solid, color: theme.background },
                        textColor: theme.textColor,
                    },
                    grid: {
                        vertLines: { color: theme.gridColor },
                        horzLines: { color: theme.gridColor },
                    },
                    timeScale: { 
                        timeVisible: true, 
                        secondsVisible: false,
                        borderVisible: false,
                    },
                    crosshair: { 
                        mode: 1 
                    },
                    rightPriceScale: {
                        borderVisible: false,
                        scaleMargins: {
                            top: 0.1,
                            bottom: 0.1,
                        },
                    },
                    leftPriceScale: {
                        visible: false,
                    },
                });
                
                this._api.timeScale().fitContent();
                
                // Call onChartReady callback
                if (onChartReady) {
                    onChartReady(this._api);
                }
            }
            return this._api;
        },
        
        free(series: ISeriesApi<SeriesType>) {
            if (this._api && series) {
                this._api.removeSeries(series);
            }
        },
        
        addPane(): number {
            if (this._api) {
                this.paneCount++;
                return this.paneCount - 1;
            }
            return 0;
        },
        
        setPaneHeight(paneIndex: number, height: number) {
            if (this._api) {
                try {
                    const panes = this._api.panes();
                    if (panes[paneIndex]) {
                        panes[paneIndex].setHeight(height);
                    }
                } catch (error) {
                    console.warn('Could not set pane height:', error);
                }
            }
        },
        
        getTotalHeight(): number {
            // Calculate total height based on number of panes
            let totalHeight = CHART_CONSTANTS.DEFAULT_MAIN_PANE_HEIGHT; // Main chart: 400px
            const additionalPanes = this.paneCount - 1;
            totalHeight += additionalPanes * CHART_CONSTANTS.INDICATOR_PANE_HEIGHT; // Each indicator pane: 150px
            totalHeight += additionalPanes * CHART_CONSTANTS.SEPARATOR_HEIGHT; // Separators: 4px each
            totalHeight += 50; // Extra spacing
            return totalHeight;
        }
    });

    // Initialize chart
    useLayoutEffect(() => {
        const currentRef = chartApiRef.current;
        const chart = currentRef.api();
        const currentApi = currentRef._api;

        const handleResize = () => {
            const newHeight = currentRef.getTotalHeight();
            chart.applyOptions({
                width: container.clientWidth,
                height: newHeight,
            });
        };

        window.addEventListener('resize', handleResize);
        return () => {
            window.removeEventListener('resize', handleResize);
            currentRef.isRemoved = true;
            if (currentApi) {
                currentApi.remove();
            }
        };
    }, [container]);

    // Update theme when it changes
    useEffect(() => {
        const currentRef = chartApiRef.current;
        if (currentRef._api) {
            currentRef._api.applyOptions({
                layout: {
                    background: { type: ColorType.Solid, color: theme.background },
                    textColor: theme.textColor,
                },
                grid: {
                    vertLines: { color: theme.gridColor },
                    horzLines: { color: theme.gridColor },
                },
            });
        }
    }, [theme]);

    // Update height when it changes
    useEffect(() => {
        const currentRef = chartApiRef.current;
        if (currentRef._api && height) {
            const newHeight = currentRef.getTotalHeight();
            currentRef._api.applyOptions({
                height: newHeight,
            });
        }
    }, [height]);

    useImperativeHandle(ref, () => chartApiRef.current.api(), []);

    return (
        <ChartContext.Provider value={chartApiRef.current}>
            {children}
        </ChartContext.Provider>
    );
});

ChartContainer.displayName = 'ChartContainer';

export { ChartContext };
