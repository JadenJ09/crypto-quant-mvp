/**
 * BacktestChart Component - Fixed Container Sizing and Pane Heights
 * Ensures proper timing of chart initialization and pane height enforcement
 */

import React, { useEffect, useRef, memo, useCallback, useState } from 'react';
import {
    createChart,
    IChartApi,
    ISeriesApi,
    Time,
    SeriesMarker,
    CandlestickData,
    CandlestickSeries,
    HistogramData,
    HistogramSeries,
    LineSeries,
    LineData,
    ColorType
} from 'lightweight-charts';

import { 
    OHLCVDataPoint, 
    IndicatorDataPoint, 
    TradeDataPoint, 
    ChartTheme,
    LIGHT_THEME,
    DARK_THEME,
    convertTime,
    CHART_CONSTANTS,
    getIndicatorPaneConfig,
    createDynamicPaneConfig
} from './types';
import styles from './BacktestChart.module.css';

interface BacktestChartProps {
    ohlcvData: OHLCVDataPoint[];
    indicatorData: Record<string, IndicatorDataPoint[]>;
    tradeData: TradeDataPoint[];
    isDarkMode: boolean;
    strategyConditions?: {
        entry_conditions: Array<{ indicator?: string; enabled: boolean }>;
        exit_conditions: Array<{ indicator?: string; enabled: boolean }>;
    };
    className?: string;
    onChartReady?: (chart: IChartApi) => void;
}

const BacktestChart: React.FC<BacktestChartProps> = memo(({
    ohlcvData,
    indicatorData,
    tradeData,
    isDarkMode,
    className = '',
    onChartReady
}) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
    const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
    const indicatorSeriesMap = useRef<Map<string, ISeriesApi<'Line' | 'Histogram'>>>(new Map());
    const resizeObserverRef = useRef<ResizeObserver | null>(null);

    // State for managing chart initialization timing
    const [isChartReady, setIsChartReady] = useState<boolean>(false);
    const [containerStabilized, setContainerStabilized] = useState<boolean>(false);

    // Get current theme
    const theme: ChartTheme = isDarkMode ? DARK_THEME : LIGHT_THEME;

    // Calculate total height needed based on indicators
    const calculateTotalHeight = useCallback(() => {
        const separatePaneIndicators = Object.keys(indicatorData).filter(indicator => {
            const config = getIndicatorPaneConfig(indicator);
            return config.target === 'separate';
        });
        
        // Group indicators to count actual panes needed
        const dynamicConfigs = createDynamicPaneConfig(separatePaneIndicators);
        const paneCount = dynamicConfigs.length;
        
        // Main chart: 400px + indicator panes: 150px each + separators: 4px each
        let totalHeight = CHART_CONSTANTS.DEFAULT_MAIN_PANE_HEIGHT;
        totalHeight += paneCount * CHART_CONSTANTS.INDICATOR_PANE_HEIGHT;
        totalHeight += paneCount * CHART_CONSTANTS.SEPARATOR_HEIGHT;
        totalHeight += 50; // Extra spacing
        
        console.log(`📐 Height calculation: Main(${CHART_CONSTANTS.DEFAULT_MAIN_PANE_HEIGHT}) + Panes(${paneCount} × ${CHART_CONSTANTS.INDICATOR_PANE_HEIGHT}) + Separators(${paneCount * CHART_CONSTANTS.SEPARATOR_HEIGHT}) + Extra(50) = ${totalHeight}px`);
        
        return totalHeight;
    }, [indicatorData]);

    // Force apply pane heights - this is the key fix for consistent heights
    const forceApplyPaneHeights = useCallback(() => {
        if (!chartRef.current || !containerStabilized) {
            console.log('🚫 Skipping height application - chart not ready or container not stabilized');
            return;
        }
        
        const chart = chartRef.current;
        const panes = chart.panes();
        
        console.log(`🎯 Forcing pane heights for ${panes.length} panes`);
        
        // Apply fixed heights: Main pane = 400px, each sub-pane = 150px
        panes.forEach((pane, index) => {
            try {
                if (index === 0) {
                    // Main price chart pane
                    pane.setHeight(CHART_CONSTANTS.DEFAULT_MAIN_PANE_HEIGHT); // 400px
                    console.log(`   📏 Pane ${index} (main): ${CHART_CONSTANTS.DEFAULT_MAIN_PANE_HEIGHT}px`);
                } else {
                    // Indicator sub-panes
                    pane.setHeight(CHART_CONSTANTS.INDICATOR_PANE_HEIGHT); // 150px
                    console.log(`   📏 Pane ${index} (indicator): ${CHART_CONSTANTS.INDICATOR_PANE_HEIGHT}px`);
                }
            } catch (error) {
                console.warn(`Failed to set height for pane ${index}:`, error);
            }
        });
    }, [containerStabilized]);

    // Create trade markers
    const createTradeMarkers = useCallback(() => {
        if (!tradeData.length) return [];
        
        const markers: SeriesMarker<Time>[] = [];
        
        tradeData.forEach(trade => {
            // Entry marker
            markers.push({
                time: convertTime(trade.time),
                position: 'belowBar',
                color: trade.side === 'Buy' ? theme.buyMarkerColor : theme.sellMarkerColor,
                shape: trade.side === 'Buy' ? 'arrowUp' : 'arrowDown',
                text: `${trade.side} @ ${trade.price.toFixed(2)}`,
                size: 1,
            });

            // Exit marker if available
            if (trade.exit_time && trade.exit_price) {
                markers.push({
                    time: convertTime(trade.exit_time),
                    position: 'aboveBar',
                    color: trade.side === 'Buy' ? theme.sellMarkerColor : theme.buyMarkerColor,
                    shape: trade.side === 'Buy' ? 'arrowDown' : 'arrowUp',
                    text: `Exit @ ${trade.exit_price.toFixed(2)}`,
                    size: 1,
                });
            }
        });
        
        return markers;
    }, [tradeData, theme]);

    // Update chart data
    const updateChartData = useCallback(() => {
        console.log('🎨 updateChartData called');
        const chart = chartRef.current;
        
        if (!chart || ohlcvData.length === 0) {
            console.log('❌ Missing requirements for chart update');
            return;
        }

        console.log('📈 Proceeding with chart data update...');

        // Clear existing indicator series
        indicatorSeriesMap.current.forEach(series => {
            chart.removeSeries(series);
        });
        indicatorSeriesMap.current.clear();

        // Update theme
        chart.applyOptions({
            layout: {
                background: { type: ColorType.Solid, color: theme.background },
                textColor: theme.textColor,
            },
            grid: {
                vertLines: { color: theme.gridColor },
                horzLines: { color: theme.gridColor },
            },
        });

        // Update candlestick series
        if (!candlestickSeriesRef.current) {
            candlestickSeriesRef.current = chart.addSeries(CandlestickSeries, {
                upColor: theme.upColor,
                downColor: theme.downColor,
                borderVisible: false,
                wickUpColor: theme.wickUpColor,
                wickDownColor: theme.wickDownColor,
                priceLineVisible: false,
                lastValueVisible: true,
                title: 'Price',
            });
        }

        const candleData: CandlestickData[] = ohlcvData.map(d => ({
            time: convertTime(d.time),
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close,
        }));

        candlestickSeriesRef.current.setData(candleData);

        // Add trade markers
        const markers = createTradeMarkers();
        if (markers.length > 0 && candlestickSeriesRef.current) {
            try {
                console.log('📍 Setting trade markers:', markers.length);
                const seriesWithMarkers = candlestickSeriesRef.current as ISeriesApi<'Candlestick'> & { 
                    setMarkers: (markers: SeriesMarker<Time>[]) => void 
                };
                seriesWithMarkers.setMarkers(markers);
            } catch (error) {
                console.warn('Could not set trade markers:', error);
            }
        }

        // Update volume series
        const hasVolumeData = ohlcvData.some(d => d.volume != null && d.volume > 0);
        if (hasVolumeData) {
            if (!volumeSeriesRef.current) {
                volumeSeriesRef.current = chart.addSeries(HistogramSeries, {
                    priceFormat: { type: 'volume' },
                    priceScaleId: 'volume_scale',
                    lastValueVisible: false,
                    priceLineVisible: false,
                    title: 'Volume',
                });
                
                chart.priceScale('volume_scale').applyOptions({
                    scaleMargins: { top: 0.7, bottom: 0 },
                });
            }

            const volumeData: HistogramData[] = ohlcvData.map(d => ({
                time: convertTime(d.time),
                value: d.volume,
                color: d.close >= d.open ? theme.volumeUpColor : theme.volumeDownColor,
            }));

            volumeSeriesRef.current.setData(volumeData);
        } else if (volumeSeriesRef.current) {
            chart.removeSeries(volumeSeriesRef.current);
            volumeSeriesRef.current = null;
        }

        // Add indicators using proper pane management
        const activeIndicators = Object.keys(indicatorData).filter(name => 
            indicatorData[name] && indicatorData[name].length > 0
        );

        if (activeIndicators.length > 0) {
            // Create dynamic pane configurations for proper grouping
            const dynamicConfigs = createDynamicPaneConfig(activeIndicators);
            console.log('📊 Dynamic pane configs:', dynamicConfigs.map(c => `${c.id}: ${c.indicators.join(', ')}`));
            
            // Add indicators to their respective panes
            dynamicConfigs.forEach((config, paneIndex) => {
                const actualPaneIndex = paneIndex + 1; // +1 because pane 0 is main chart
                
                config.indicators.forEach((indicatorName, seriesIndex) => {
                    const data = indicatorData[indicatorName];
                    if (data && data.length > 0) {
                        const validData = data.filter(d => d.value != null && !isNaN(d.value));
                        
                        if (validData.length > 0) {
                            const lineData: LineData[] = validData.map(point => ({
                                time: convertTime(point.time),
                                value: point.value
                            }));

                            // Choose color based on series index within the pane
                            const color = theme.indicatorColors[seriesIndex % theme.indicatorColors.length];
                            
                            const series = chart.addSeries(LineSeries, {
                                color,
                                lineWidth: 2,
                                priceFormat: {
                                    type: 'price',
                                    precision: 4,
                                    minMove: 0.0001,
                                },
                                priceLineVisible: false,
                                lastValueVisible: false,
                                title: indicatorName,
                            }, actualPaneIndex);

                            series.setData(lineData);
                            indicatorSeriesMap.current.set(indicatorName, series);
                            
                            console.log(`   📈 Added ${indicatorName} to pane ${actualPaneIndex} with color ${color}`);
                        }
                    }
                });
            });
        }

        // Fit content to show all data
        chart.timeScale().fitContent();

        // Calculate and apply new height
        const newHeight = calculateTotalHeight();
        if (chartContainerRef.current) {
            const width = chartContainerRef.current.clientWidth;
            chart.resize(width, newHeight);
        }
        
        // CRITICAL: Apply pane heights AFTER container has stabilized and data is loaded
        // Use multiple timeouts with escalating delays for robustness
        setTimeout(() => forceApplyPaneHeights(), 100);
        setTimeout(() => forceApplyPaneHeights(), 300);
        setTimeout(() => forceApplyPaneHeights(), 600);
        setTimeout(() => forceApplyPaneHeights(), 1000);
        
        console.log('✅ Chart data update completed', { newHeight, activeIndicators: Object.keys(indicatorData).length });
    }, [ohlcvData, indicatorData, theme, calculateTotalHeight, createTradeMarkers, forceApplyPaneHeights]);

    // Container stabilization detection effect
    useEffect(() => {
        if (!chartContainerRef.current) return;

        let stabilizationTimer: number;
        let lastWidth = 0;
        let lastHeight = 0;
        let checkCount = 0;

        const checkStabilization = () => {
            if (!chartContainerRef.current) return;

            const currentWidth = chartContainerRef.current.clientWidth;
            const currentHeight = chartContainerRef.current.clientHeight;
            
            console.log(`📏 Container size check ${checkCount}: ${currentWidth}x${currentHeight}`);

            if (currentWidth === lastWidth && currentHeight === lastHeight && currentWidth > 0) {
                // Container has stabilized
                console.log('✅ Container stabilized at:', currentWidth, 'x', currentHeight);
                setContainerStabilized(true);
                return;
            }

            lastWidth = currentWidth;
            lastHeight = currentHeight;
            checkCount++;

            // Continue checking for stabilization (max 20 checks = 2 seconds)
            if (checkCount < 20) {
                stabilizationTimer = setTimeout(checkStabilization, 100);
            } else {
                // Force stabilization after timeout
                console.log('⚠️ Forcing container stabilization after timeout');
                setContainerStabilized(true);
            }
        };

        // Start checking after a brief delay
        stabilizationTimer = setTimeout(checkStabilization, 50);

        return () => {
            if (stabilizationTimer) {
                clearTimeout(stabilizationTimer);
            }
        };
    }, []);

    // Initialize chart on mount - only once
    useEffect(() => {
        if (!chartContainerRef.current || chartRef.current) return;

        console.log('🎯 Initializing chart...');
        const initialHeight = calculateTotalHeight();
        
        const chart = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
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
        
        // Mark chart as ready
        setIsChartReady(true);

        // Setup resize observer
        if (resizeObserverRef.current) {
            resizeObserverRef.current.disconnect();
        }

        resizeObserverRef.current = new ResizeObserver(entries => {
            const entry = entries[0];
            if (entry && chart) {
                const { width } = entry.contentRect;
                const currentHeight = calculateTotalHeight();
                chart.resize(width, currentHeight);
                
                // Re-apply pane heights after resize
                setTimeout(() => forceApplyPaneHeights(), 100);
            }
        });

        resizeObserverRef.current.observe(chartContainerRef.current);

        // Notify parent component
        if (onChartReady) {
            onChartReady(chart);
        }
        
        console.log('✅ Chart initialized successfully');
        
        return () => {
            console.log('🧹 Cleaning up chart...');
            // Copy ref values to variables to avoid React warning
            const currentIndicatorMap = indicatorSeriesMap.current;
            
            if (resizeObserverRef.current) {
                resizeObserverRef.current.disconnect();
            }
            if (chartRef.current) {
                chartRef.current.remove();
                chartRef.current = null;
                candlestickSeriesRef.current = null;
                volumeSeriesRef.current = null;
                currentIndicatorMap.clear();
            }
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); // Empty dependency array - initialize only once

    // Update chart when data changes - ONLY when container is stabilized
    useEffect(() => {
        console.log('📊 Chart data effect triggered', { 
            hasData: ohlcvData.length > 0, 
            hasChart: !!chartRef.current,
            isReady: isChartReady,
            containerStabilized,
            ohlcvLength: ohlcvData.length 
        });
        
        if (ohlcvData.length > 0 && chartRef.current && isChartReady && containerStabilized) {
            console.log('🔄 Updating chart data (container stabilized)...');
            // Small delay to ensure DOM has fully rendered
            setTimeout(() => {
                updateChartData();
            }, 100);
        }
    }, [ohlcvData, indicatorData, tradeData, updateChartData, isChartReady, containerStabilized]);

    return (
        <div className={`${styles.chartContainer} ${className}`}>
            <div 
                ref={chartContainerRef} 
                className={styles.chartWrapper}
                style={{ 
                    height: calculateTotalHeight(),
                    minHeight: calculateTotalHeight(),
                    width: '100%'
                }}
                data-chart-container
            />
        </div>
    );
});

BacktestChart.displayName = 'BacktestChart';

export default BacktestChart;
export { BacktestChart };
export type { BacktestChartProps };
