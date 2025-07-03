import React, { useEffect, useRef, memo } from 'react';
import {
    createChart,
    IChartApi,
    ISeriesApi,
    UTCTimestamp,
    Time,
    SeriesMarker,
    LineData,
    HistogramData,
    CandlestickData,
    CandlestickSeries,
    HistogramSeries,
    LineSeries,
    createSeriesMarkers,
    ColorType
} from 'lightweight-charts';

// Types for the chart data
export interface OHLCVDataPoint {
    time: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

export interface IndicatorDataPoint {
    time: string;
    value: number;
}

export interface TradeDataPoint {
    time: string;
    side: 'Buy' | 'Sell';
    price: number;
    quantity: number;
}

interface BacktestChartProps {
    ohlcvData: OHLCVDataPoint[];
    indicatorData: Record<string, IndicatorDataPoint[]>;
    tradeData: TradeDataPoint[];
    isDarkMode: boolean;
    strategyConditions?: {
        entry_conditions: Array<{ indicator?: string; enabled: boolean }>;
        exit_conditions: Array<{ indicator?: string; enabled: boolean }>;
    };
}

// Chart themes
const lightThemeColors = {
    background: '#FFFFFF',
    textColor: '#333333',
    gridColor: '#E6E6E6',
    upColor: '#26A69A',
    downColor: '#EF5350',
    wickUpColor: '#26A69A',
    wickDownColor: '#EF5350',
    volumeUpColor: 'rgba(38, 166, 154, 0.5)',
    volumeDownColor: 'rgba(239, 83, 80, 0.5)',
    buyMarkerColor: '#2196F3',
    sellMarkerColor: '#FF9800',
    indicatorColors: ['#2962FF', '#FF6D00', '#2E7D32', '#D50000', '#5D4037'],
    rsiLineColor: '#FF6D00',
    rsiLevelColor: '#787B86',
};

const darkThemeColors = {
    background: 'hsl(200, 25%, 8%)', // Exact match to CSS variable --background
    textColor: 'hsl(180, 10%, 95%)', // Exact match to CSS variable --foreground
    gridColor: 'hsl(200, 20%, 25%)', // Exact match to CSS variable --border
    upColor: '#26A69A',
    downColor: '#EF5350',
    wickUpColor: '#26A69A',
    wickDownColor: '#EF5350',
    volumeUpColor: 'rgba(38, 166, 154, 0.5)',
    volumeDownColor: 'rgba(239, 83, 80, 0.5)',
    buyMarkerColor: '#2196F3',
    sellMarkerColor: '#FF9800',
    indicatorColors: ['#58A6FF', '#F78166', '#7EE787', '#FF7B72', '#FFA657'],
    rsiLineColor: '#F78166',
    rsiLevelColor: 'hsl(180, 10%, 65%)', // Exact match to CSS variable --muted-foreground
};

// Helper function to check if an indicator should be in a separate pane
const shouldUseSeparatePane = (indicatorName: string): boolean => {
    const nameUpper = indicatorName.toUpperCase();
    return nameUpper.includes('RSI') || 
           nameUpper.includes('MACD') || 
           nameUpper.includes('HISTOGRAM') || 
           nameUpper.includes('SIGNAL') || 
           nameUpper.includes('STOCH');
};

// Convert string time to timestamp for the chart
const convertTime = (timeStr: string): UTCTimestamp => {
    return (new Date(timeStr).getTime() / 1000) as UTCTimestamp;
};

const BacktestChart: React.FC<BacktestChartProps> = memo(({
    ohlcvData,
    indicatorData,
    tradeData,
    isDarkMode,
}) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRefs = useRef<Record<string, ISeriesApi<'Line' | 'Candlestick' | 'Histogram'>>>({});
    const markersRef = useRef<{ setMarkers?: (markers: never[]) => void } | null>(null);

    // Calculate dynamic height based on indicators
    const calculateChartHeight = () => {
        const mainChartHeight = 400;
        const subPaneHeight = 150; // Fixed sub-pane height
        const indicatorCount = Object.keys(indicatorData).filter(name => shouldUseSeparatePane(name)).length;
        
        // Total height is main chart + all sub-panes + separators
        const separatorHeight = indicatorCount > 0 ? indicatorCount * 4 : 0; // 4px per separator
        const totalHeight = mainChartHeight + (indicatorCount * subPaneHeight) + separatorHeight;
        
        return totalHeight;
    };

    const totalChartHeight = calculateChartHeight();

    // Initialize chart
    useEffect(() => {
        if (!chartContainerRef.current) return;

        const theme = isDarkMode ? darkThemeColors : lightThemeColors;
        
        const chart = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
            height: totalChartHeight,
            layout: {
                background: { type: ColorType.Solid, color: theme.background },
                textColor: theme.textColor,
                panes: {
                    separatorColor: theme.gridColor,
                    separatorHoverColor: theme.gridColor,
                    enableResize: false,
                },
            },
            grid: {
                vertLines: { color: theme.gridColor },
                horzLines: { color: theme.gridColor },
            },
            timeScale: { 
                timeVisible: true, 
                secondsVisible: false,
                borderVisible: false,
                fixLeftEdge: false,
                fixRightEdge: false
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

        chartRef.current = chart;

        // Handle resize with proper pane height management
        const resizeObserver = new ResizeObserver(entries => {
            const entry = entries[0];
            if (entry && chart) {
                const { width } = entry.contentRect;
                chart.resize(width, totalChartHeight);
                
                // Ensure pane heights are maintained after resize
                setTimeout(() => {
                    const panes = chart.panes();
                    if (panes[0]) panes[0].setHeight(400); // Main price chart
                    for (let i = 1; i < panes.length; i++) {
                        if (panes[i]) panes[i].setHeight(150); // Sub-panes
                    }
                }, 0);
            }
        });

        resizeObserver.observe(chartContainerRef.current);

        return () => {
            resizeObserver.disconnect();
            if (chartRef.current) {
                chartRef.current.remove();
                chartRef.current = null;
                seriesRefs.current = {};
                markersRef.current = null;
            }
        };
    }, [isDarkMode, totalChartHeight]);        // Update data
    useEffect(() => {
        const chart = chartRef.current;
        if (!chart || ohlcvData.length === 0) return;

        const theme = isDarkMode ? darkThemeColors : lightThemeColors;
        
        // Update chart height if it changed
        if (chart) {
            const containerWidth = chartContainerRef.current?.clientWidth || 800;
            chart.resize(containerWidth, totalChartHeight);
        }

        // Update chart theme
        chart.applyOptions({
            layout: {
                background: { type: ColorType.Solid, color: theme.background },
                textColor: theme.textColor,
                panes: {
                    separatorColor: theme.gridColor,
                    separatorHoverColor: theme.gridColor,
                    enableResize: false,
                },
            },
            grid: {
                vertLines: { color: theme.gridColor },
                horzLines: { color: theme.gridColor },
            },
        });

        // Clear existing markers
        if (markersRef.current && markersRef.current.setMarkers) {
            markersRef.current.setMarkers([]);
            markersRef.current = null;
        }

        // Function to setup pane heights properly using PaneAPI
        const setupPaneHeights = () => {
            const panes = chart.panes();
            
            // Set main pane (price chart) to 400px
            if (panes[0]) {
                panes[0].setHeight(400);
            }
            
            // Set all sub-panes to 150px each
            for (let i = 1; i < panes.length; i++) {
                if (panes[i]) {
                    panes[i].setHeight(150);
                }
            }
        };

        // Candlestick series
        if (!seriesRefs.current['candlestick']) {
            seriesRefs.current['candlestick'] = chart.addSeries(CandlestickSeries, {
                upColor: theme.upColor,
                downColor: theme.downColor,
                borderVisible: false,
                wickUpColor: theme.wickUpColor,
                wickDownColor: theme.wickDownColor,
            });
        } else {
            (seriesRefs.current['candlestick'] as ISeriesApi<'Candlestick'>).applyOptions({
                upColor: theme.upColor,
                downColor: theme.downColor,
                wickUpColor: theme.wickUpColor,
                wickDownColor: theme.wickDownColor,
            });
        }

        const candleData: CandlestickData[] = ohlcvData.map(d => ({
            time: convertTime(d.time),
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close,
        }));

        seriesRefs.current['candlestick'].setData(candleData);

        // Volume series
        const hasVolumeData = ohlcvData.some(d => d.volume != null && d.volume > 0);
        if (hasVolumeData) {
            if (!seriesRefs.current['volume']) {
                seriesRefs.current['volume'] = chart.addSeries(HistogramSeries, {
                    priceFormat: { type: 'volume' },
                    priceScaleId: 'volume_scale',
                    lastValueVisible: false,
                    priceLineVisible: false,
                });
                chart.priceScale('volume_scale').applyOptions({
                    scaleMargins: { top: 0.8, bottom: 0 },
                });
            }

            const volumeData: HistogramData[] = ohlcvData.map(d => ({
                time: convertTime(d.time),
                value: d.volume,
                color: d.close >= d.open ? theme.volumeUpColor : theme.volumeDownColor,
            }));

            seriesRefs.current['volume'].setData(volumeData);
        } else if (seriesRefs.current['volume']) {
            chart.removeSeries(seriesRefs.current['volume']);
            delete seriesRefs.current['volume'];
        }

        // Indicators
        const existingIndicators = Object.keys(seriesRefs.current).filter(k => k.startsWith('indicator_'));
        const currentIndicators = Object.keys(indicatorData).map(k => `indicator_${k}`);

        // Remove old indicators
        existingIndicators.forEach(key => {
            if (!currentIndicators.includes(key)) {
                chart.removeSeries(seriesRefs.current[key]);
                delete seriesRefs.current[key];
            }
        });

        // Add/update indicators
        let colorIndex = 0;
        let nextPaneIndex = 1; // Start with pane 1 for indicators
        
        Object.entries(indicatorData).forEach(([name, data]) => {
            const key = `indicator_${name}`;
            const validData = data.filter(d => d.value != null && !isNaN(d.value));
            
            if (validData.length === 0) {
                if (seriesRefs.current[key]) {
                    chart.removeSeries(seriesRefs.current[key]);
                    delete seriesRefs.current[key];
                }
                // Also remove reference lines for RSI if they exist
                const overboughtKey = `${key}_overbought`;
                const oversoldKey = `${key}_oversold`;
                if (seriesRefs.current[overboughtKey]) {
                    chart.removeSeries(seriesRefs.current[overboughtKey]);
                    delete seriesRefs.current[overboughtKey];
                }
                if (seriesRefs.current[oversoldKey]) {
                    chart.removeSeries(seriesRefs.current[oversoldKey]);
                    delete seriesRefs.current[oversoldKey];
                }
                return;
            }

            // Check if this indicator should be in a separate pane
            if (shouldUseSeparatePane(name)) {
                console.log(`Adding indicator: ${name} to separate pane ${nextPaneIndex}`);
                
                const indicatorData: LineData[] = validData.map(point => ({
                    time: convertTime(point.time),
                    value: point.value
                }));
                
                // Determine color based on indicator type
                let indicatorColor = theme.indicatorColors[colorIndex % theme.indicatorColors.length];
                if (name.toUpperCase().includes('RSI')) {
                    indicatorColor = theme.rsiLineColor;
                } else if (name.toUpperCase().includes('MACD')) {
                    indicatorColor = '#2962FF'; // Blue for MACD
                } else if (name.toUpperCase().includes('SIGNAL')) {
                    indicatorColor = '#FF6D00'; // Orange for signal
                } else if (name.toUpperCase().includes('HISTOGRAM')) {
                    indicatorColor = '#FF7043'; // Red-orange for histogram
                }
                
                if (!seriesRefs.current[key]) {
                    // Determine if this should be a histogram or line series
                    const isHistogram = name.toUpperCase().includes('HISTOGRAM') || name.toUpperCase().includes('MACD_HIST');
                    
                    if (isHistogram) {
                        // Add as histogram series
                        seriesRefs.current[key] = chart.addSeries(HistogramSeries, {
                            color: indicatorColor,
                            priceFormat: {
                                type: 'price',
                                precision: 4,
                                minMove: 0.0001,
                            },
                            priceLineVisible: false,
                            lastValueVisible: false,
                            title: name,
                        }, nextPaneIndex);
                        
                        // Convert line data to histogram data
                        const histogramData: HistogramData[] = validData.map(point => ({
                            time: convertTime(point.time),
                            value: point.value,
                            color: point.value >= 0 ? '#26A69A' : '#EF5350' // Green for positive, red for negative
                        }));
                        
                        seriesRefs.current[key].setData(histogramData);
                    } else {
                        // Add as line series
                        seriesRefs.current[key] = chart.addSeries(LineSeries, {
                            color: indicatorColor,
                            lineWidth: 2,
                            priceFormat: {
                                type: 'price',
                                precision: 4,
                                minMove: 0.0001,
                            },
                            priceLineVisible: false,
                            lastValueVisible: false,
                            title: name,
                        }, nextPaneIndex);
                        
                        seriesRefs.current[key].setData(indicatorData);
                    }
                    
                    // Add RSI reference lines (70, 30) for RSI indicator
                    if (name.toUpperCase().includes('RSI')) {
                        // Create reference lines for overbought (70) and oversold (30)
                        const timeRange = validData.map(point => convertTime(point.time));
                        
                        // Overbought line (70)
                        const overboughtKey = `${key}_overbought`;
                        if (!seriesRefs.current[overboughtKey]) {
                            seriesRefs.current[overboughtKey] = chart.addSeries(LineSeries, {
                                color: '#ff4444',
                                lineWidth: 1,
                                lineStyle: 2, // dashed line
                                priceLineVisible: false,
                                lastValueVisible: false,
                                title: 'Overbought (70)',
                            }, nextPaneIndex); // Same pane as RSI
                            
                            const overboughtData = timeRange.map(time => ({ time, value: 70 }));
                            seriesRefs.current[overboughtKey].setData(overboughtData);
                        }
                        
                        // Oversold line (30)
                        const oversoldKey = `${key}_oversold`;
                        if (!seriesRefs.current[oversoldKey]) {
                            seriesRefs.current[oversoldKey] = chart.addSeries(LineSeries, {
                                color: '#44ff44',
                                lineWidth: 1,
                                lineStyle: 2, // dashed line
                                priceLineVisible: false,
                                lastValueVisible: false,
                                title: 'Oversold (30)',
                            }, nextPaneIndex); // Same pane as RSI
                            
                            const oversoldData = timeRange.map(time => ({ time, value: 30 }));
                            seriesRefs.current[oversoldKey].setData(oversoldData);
                        }
                    }
                    
                    // Add zero line for MACD histogram
                    if (name.toUpperCase().includes('MACD') && name.toUpperCase().includes('HISTOGRAM')) {
                        const zeroLineKey = `${key}_zero`;
                        if (!seriesRefs.current[zeroLineKey]) {
                            const timeRange = validData.map(point => convertTime(point.time));
                            seriesRefs.current[zeroLineKey] = chart.addSeries(LineSeries, {
                                color: '#787B86',
                                lineWidth: 1,
                                lineStyle: 2, // dashed line
                                priceLineVisible: false,
                                lastValueVisible: false,
                                title: 'Zero Line',
                            }, nextPaneIndex);
                            
                            const zeroData = timeRange.map(time => ({ time, value: 0 }));
                            seriesRefs.current[zeroLineKey].setData(zeroData);
                        }
                    }
                    
                    // Set pane height immediately after series creation
                    setTimeout(() => {
                        const panes = chart.panes();
                        if (panes[nextPaneIndex]) {
                            panes[nextPaneIndex].setHeight(150);
                        }
                    }, 0);
                    
                    nextPaneIndex++;
                } else {
                    // Update existing series
                    if (name.toUpperCase().includes('HISTOGRAM') || name.toUpperCase().includes('MACD_HIST')) {
                        const histogramData: HistogramData[] = validData.map(point => ({
                            time: convertTime(point.time),
                            value: point.value,
                            color: point.value >= 0 ? '#26A69A' : '#EF5350'
                        }));
                        seriesRefs.current[key].setData(histogramData);
                    } else {
                        seriesRefs.current[key].setData(indicatorData);
                    }
                }
                return;
            }

            // For indicators that should stay on the main chart
            const color = theme.indicatorColors[colorIndex % theme.indicatorColors.length];
            colorIndex++;

            const lineData: LineData[] = validData.map(d => ({
                time: convertTime(d.time),
                value: d.value,
            }));

            if (!seriesRefs.current[key]) {
                seriesRefs.current[key] = chart.addSeries(LineSeries, {
                    color: color,
                    lineWidth: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    priceScaleId: 'right', // Use main price scale
                    title: name,
                });
            } else {
                (seriesRefs.current[key] as ISeriesApi<'Line'>).applyOptions({
                    color: color,
                });
            }

            seriesRefs.current[key].setData(lineData);
        });

        // Trade markers using the new v5 API
        const candleSeries = seriesRefs.current['candlestick'] as ISeriesApi<'Candlestick'>;
        if (candleSeries && tradeData.length > 0) {
            try {
                const markers: SeriesMarker<Time>[] = tradeData.map(trade => ({
                    time: convertTime(trade.time),
                    position: trade.side === 'Buy' ? 'belowBar' : 'aboveBar',
                    color: trade.side === 'Buy' ? theme.buyMarkerColor : theme.sellMarkerColor,
                    shape: trade.side === 'Buy' ? 'arrowUp' : 'arrowDown',
                    text: `${trade.side[0]} @ ${trade.price.toFixed(2)}`,
                    size: 1,
                }));

                markersRef.current = createSeriesMarkers(candleSeries, markers) as unknown as { setMarkers?: (markers: never[]) => void };
            } catch (error) {
                console.warn('Could not create markers:', error);
                // Fallback: try the old API if available
                try {
                    if ('setMarkers' in candleSeries) {
                        const markers: SeriesMarker<Time>[] = tradeData.map(trade => ({
                            time: convertTime(trade.time),
                            position: trade.side === 'Buy' ? 'belowBar' : 'aboveBar',
                            color: trade.side === 'Buy' ? theme.buyMarkerColor : theme.sellMarkerColor,
                            shape: trade.side === 'Buy' ? 'arrowUp' : 'arrowDown',
                            text: `${trade.side[0]} @ ${trade.price.toFixed(2)}`,
                            size: 1,
                        }));
                        (candleSeries as unknown as { setMarkers: (markers: SeriesMarker<Time>[]) => void }).setMarkers(markers);
                    }
                } catch {
                    console.warn('Markers not supported in this version of lightweight-charts');
                }
            }
        }

        // Fit content to show all data
        chart.timeScale().fitContent();

        // Setup pane heights after all indicators are added
        // Use setTimeout to ensure all DOM updates are complete
        setTimeout(() => {
            setupPaneHeights();
        }, 0);

    }, [ohlcvData, indicatorData, tradeData, isDarkMode, totalChartHeight]);

    return (
        <div 
            ref={chartContainerRef} 
            className="w-full rounded-md border border-gray-200 dark:border-gray-800 chart-container chart-dynamic-height"
            style={{ height: `${totalChartHeight}px` }}
        />
    );
});

BacktestChart.displayName = 'BacktestChart';

export default BacktestChart;
