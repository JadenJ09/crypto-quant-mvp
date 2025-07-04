/**
 * Pane Manager
 * Handles all pane-related operations including creation, management, and layout
 */

import { 
    IChartApi, 
    ISeriesApi, 
    LineSeries, 
    HistogramSeries,
    LineData,
    HistogramData,
    SeriesType
} from 'lightweight-charts';

import { 
    ChartTheme, 
    PaneConfiguration, 
    CHART_CONSTANTS,
    getReferenceLines,
    convertTime,
    IndicatorDataPoint,
    createDynamicPaneConfig,
    DynamicPaneConfig,
    getIndicatorPaneConfig
} from './types';

export class PaneManager {
    private chart: IChartApi;
    private theme: ChartTheme;
    private paneConfigurations: Map<string, PaneConfiguration> = new Map();
    private dynamicPanes: Map<string, DynamicPaneConfig> = new Map();
    private seriesMap: Map<string, ISeriesApi<SeriesType>> = new Map();
    private referenceLineMap: Map<string, ISeriesApi<SeriesType>[]> = new Map();
    private nextPaneIndex = 1; // Start with 1 since 0 is the main pane

    constructor(chart: IChartApi, theme: ChartTheme) {
        this.chart = chart;
        this.theme = theme;
        this.initializeMainPane();
    }

    private initializeMainPane(): void {
        const mainPane: PaneConfiguration = {
            id: 'main',
            height: CHART_CONSTANTS.DEFAULT_MAIN_PANE_HEIGHT, // Fixed 400px
            indicators: ['candlestick', 'volume'],
        };
        this.paneConfigurations.set('main', mainPane);
    }

    /**
     * Configure panes dynamically based on active indicators
     * This replaces the old individual indicator adding approach
     */
    public configurePanesForIndicators(indicators: string[]): void {
        // Clear existing dynamic panes
        this.clearDynamicPanes();
        
        // Create dynamic pane configurations
        const dynamicConfigs = createDynamicPaneConfig(indicators);
        
        // Set up each dynamic pane
        dynamicConfigs.forEach(config => {
            this.dynamicPanes.set(config.id, config);
        });
        
        // Apply pane heights
        this.applyPaneHeights();
    }

    /**
     * Clear all dynamic panes (keeps main pane)
     */
    private clearDynamicPanes(): void {
        // Remove all dynamic pane series
        for (const [key, series] of this.seriesMap) {
            if (key.startsWith('indicator_')) {
                this.chart.removeSeries(series);
            }
        }

        // Clear reference lines from dynamic panes
        for (const lines of this.referenceLineMap.values()) {
            lines.forEach(line => this.chart.removeSeries(line));
        }

        // Clear dynamic pane data
        this.dynamicPanes.clear();
        this.referenceLineMap.clear();
        
        // Remove indicator series from map
        const indicatorKeys = Array.from(this.seriesMap.keys()).filter(key => key.startsWith('indicator_'));
        indicatorKeys.forEach(key => this.seriesMap.delete(key));
    }

    /**
     * Get overlay indicators (those that should be displayed on main chart)
     */
    public getOverlayIndicators(indicators: string[]): string[] {
        return indicators.filter(indicator => {
            const config = getIndicatorPaneConfig(indicator);
            return config.target === 'main';
        });
    }

    /**
     * Get separate pane indicators (those that need their own panes)
     */
    public getSeparatePaneIndicators(indicators: string[]): string[] {
        return indicators.filter(indicator => {
            const config = getIndicatorPaneConfig(indicator);
            return config.target === 'separate';
        });
    }

    /**
     * Add indicator data to the appropriate pane (new dynamic approach)
     */
    public addIndicatorToDynamicPane(
        indicatorName: string, 
        data: IndicatorDataPoint[], 
        colorIndex: number = 0
    ): void {
        // Find which dynamic pane contains this indicator
        const targetPane = Array.from(this.dynamicPanes.values())
            .find(pane => pane.indicators.includes(indicatorName));
        
        if (!targetPane) {
            console.warn(`No pane found for indicator: ${indicatorName}`);
            return;
        }

        // Create the indicator series in the appropriate pane
        this.createIndicatorSeries(indicatorName, data, targetPane.id, colorIndex);
        
        // Add reference lines if needed
        this.addReferenceLines(indicatorName, targetPane.id, data);
    }

    /**
     * Add an indicator to the appropriate pane (main overlay or separate pane)
     */
    public addIndicator(
        indicatorName: string, 
        data: IndicatorDataPoint[], 
        colorIndex: number = 0
    ): void {
        const config = getIndicatorPaneConfig(indicatorName);
        
        if (config.target === 'main') {
            // Add as overlay on main chart
            this.addOverlayIndicator(indicatorName, data, colorIndex);
        } else {
            // Add to separate pane
            this.addIndicatorToDynamicPane(indicatorName, data, colorIndex);
        }
    }

    /**
     * Add overlay indicator to the main chart (pane 0)
     */
    public addOverlayIndicator(
        indicatorName: string, 
        data: IndicatorDataPoint[], 
        colorIndex: number = 0
    ): void {
        const seriesKey = `indicator_${indicatorName}`;
        const validData = data.filter(d => d.value != null && !isNaN(d.value));
        
        if (validData.length === 0) return;

        const color = this.getIndicatorColor(indicatorName, colorIndex);
        
        // Add to main chart (pane 0)
        const series = this.chart.addSeries(LineSeries, {
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
        }); // No paneIndex = main chart (pane 0)

        const lineData: LineData[] = validData.map(point => ({
            time: convertTime(point.time),
            value: point.value
        }));

        series.setData(lineData);
        this.seriesMap.set(seriesKey, series);
    }

    /**
     * Remove an indicator and cleanup its pane if empty
     */
    public removeIndicator(indicatorName: string): void {
        const seriesKey = `indicator_${indicatorName}`;
        const series = this.seriesMap.get(seriesKey);
        
        if (series) {
            this.chart.removeSeries(series);
            this.seriesMap.delete(seriesKey);
        }

        // Remove reference lines
        this.removeReferenceLines(indicatorName);

        // Check if pane should be removed
        this.cleanupEmptyPanes();
    }

    /**
     * Update existing indicator data
     */
    public updateIndicator(indicatorName: string, data: IndicatorDataPoint[]): void {
        const seriesKey = `indicator_${indicatorName}`;
        const series = this.seriesMap.get(seriesKey);
        
        if (series && data.length > 0) {
            const validData = data.filter(d => d.value != null && !isNaN(d.value));
            
            if (validData.length > 0) {
                const isHistogram = this.isHistogramIndicator(indicatorName);
                
                if (isHistogram) {
                    const histogramData: HistogramData[] = validData.map(point => ({
                        time: convertTime(point.time),
                        value: point.value,
                        color: point.value >= 0 ? this.theme.upColor : this.theme.downColor
                    }));
                    (series as ISeriesApi<'Histogram'>).setData(histogramData);
                } else {
                    const lineData: LineData[] = validData.map(point => ({
                        time: convertTime(point.time),
                        value: point.value
                    }));
                    (series as ISeriesApi<'Line'>).setData(lineData);
                }

                // Update reference lines
                this.updateReferenceLines(indicatorName, validData);
            }
        }
    }

    /**
     * Get the total height needed for the chart container (dynamic based on panes)
     */
    public getTotalHeight(): number {
        // Fixed main price chart: 400px (always)
        let totalHeight = CHART_CONSTANTS.DEFAULT_MAIN_PANE_HEIGHT;
        
        // Add fixed height for each indicator pane: 150px per pane
        const separatePaneCount = this.dynamicPanes.size;
        totalHeight += separatePaneCount * CHART_CONSTANTS.INDICATOR_PANE_HEIGHT;
        
        // Add separator heights
        const separatorHeight = separatePaneCount * CHART_CONSTANTS.SEPARATOR_HEIGHT;
        
        // Add extra 130px for better visual spacing
        const extraSpacing = 130;
        
        return totalHeight + separatorHeight + extraSpacing;
    }

    /**
     * Apply fixed pane heights to the chart
     */
    public applyPaneHeights(): void {
        // Use setTimeout to ensure DOM is ready and then apply heights multiple times for robustness
        setTimeout(() => {
            this.forceApplyPaneHeights();
        }, 0);
        
        setTimeout(() => {
            this.forceApplyPaneHeights();
        }, 100);
        
        setTimeout(() => {
            this.forceApplyPaneHeights();
        }, 500);
    }

    /**
     * Force apply pane heights immediately
     */
    private forceApplyPaneHeights(): void {
        const chartPanes = this.chart.panes();
        
        // Main pane: Fixed 400px (always) - pane index 0
        if (chartPanes[0]) {
            chartPanes[0].setHeight(CHART_CONSTANTS.DEFAULT_MAIN_PANE_HEIGHT);
            console.log(`📏 Applied main pane height: ${CHART_CONSTANTS.DEFAULT_MAIN_PANE_HEIGHT}px`);
        }
        
        // Indicator panes: Fixed 150px each - pane index 1, 2, 3, etc.
        for (let paneIndex = 1; paneIndex < chartPanes.length; paneIndex++) {
            if (chartPanes[paneIndex]) {
                chartPanes[paneIndex].setHeight(CHART_CONSTANTS.INDICATOR_PANE_HEIGHT);
                console.log(`📏 Applied indicator pane ${paneIndex} height: ${CHART_CONSTANTS.INDICATOR_PANE_HEIGHT}px`);
            }
        }
    }

    /**
     * Clear all indicators and reset panes (updated for dynamic panes)
     */
    public clearAllIndicators(): void {
        // Clear dynamic panes first
        this.clearDynamicPanes();

        // Clear any remaining legacy series
        for (const [key, series] of this.seriesMap) {
            if (key.startsWith('indicator_')) {
                this.chart.removeSeries(series);
            }
        }

        // Clear reference lines
        for (const lines of this.referenceLineMap.values()) {
            lines.forEach(line => this.chart.removeSeries(line));
        }

        // Clear maps
        this.seriesMap.clear();
        this.referenceLineMap.clear();

        // Reset pane configurations (keep main pane)
        this.paneConfigurations.clear();
        this.initializeMainPane();
        this.nextPaneIndex = 1;
    }

    /**
     * Update theme for all series
     */
    public updateTheme(newTheme: ChartTheme): void {
        this.theme = newTheme;
        
        // Update all series colors
        for (const [key, series] of this.seriesMap) {
            if (key.startsWith('indicator_')) {
                const indicatorName = key.replace('indicator_', '');
                const colorIndex = this.getColorIndex(indicatorName);
                const color = this.getIndicatorColor(indicatorName, colorIndex);
                
                if (series.seriesType() === 'Line') {
                    (series as ISeriesApi<'Line'>).applyOptions({ color });
                } else if (series.seriesType() === 'Histogram') {
                    (series as ISeriesApi<'Histogram'>).applyOptions({ color });
                }
            }
        }

        // Update reference lines
        for (const [indicatorName, lines] of this.referenceLineMap) {
            const refLines = getReferenceLines(indicatorName);
            lines.forEach((line, index) => {
                if (refLines[index]) {
                    (line as ISeriesApi<'Line'>).applyOptions({ 
                        color: refLines[index].color 
                    });
                }
            });
        }
    }

    /**
     * Refresh chart after re-rendering to prevent blank chart issues
     * This method is called 0.2 seconds after chart updates to ensure proper rendering
     */
    public refreshChart(): void {
        setTimeout(() => {
            try {
                // Force chart to re-render by fitting content
                this.chart.timeScale().fitContent();
                
                // Apply pane heights again to ensure proper layout
                this.applyPaneHeights();
                
                // Force chart resize with current dimensions
                const chartPanes = this.chart.panes();
                if (chartPanes.length > 0) {
                    const totalHeight = this.getTotalHeight();
                    
                    // Try to get chart container element for width
                    const chartElement = document.querySelector('[data-chart-container]') as HTMLElement;
                    if (chartElement) {
                        const width = chartElement.clientWidth;
                        if (width > 0) {
                            this.chart.resize(width, totalHeight);
                        }
                    } else {
                        // Fallback: use default width and force resize
                        this.chart.resize(800, totalHeight);
                    }
                    
                    // Additional refresh steps to ensure chart renders properly
                    setTimeout(() => {
                        this.chart.timeScale().fitContent();
                    }, 50); // Small additional delay for final content fit
                }
            } catch (error) {
                console.warn('Could not refresh chart:', error);
            }
        }, 200); // 0.2 seconds as requested
    }

    private createIndicatorSeries(
        indicatorName: string, 
        data: IndicatorDataPoint[], 
        paneId: string, 
        colorIndex: number
    ): void {
        const seriesKey = `indicator_${indicatorName}`;
        const validData = data.filter(d => d.value != null && !isNaN(d.value));
        
        if (validData.length === 0) return;

        // Use the official TradingView pane API
        const paneIndex = this.getPaneIndex(paneId);
        const color = this.getIndicatorColor(indicatorName, colorIndex);
        const isHistogram = this.isHistogramIndicator(indicatorName);

        if (isHistogram) {
            // Add histogram series to specified pane using official API
            const series = this.chart.addSeries(HistogramSeries, {
                color,
                priceFormat: {
                    type: 'price',
                    precision: 4,
                    minMove: 0.0001,
                },
                priceLineVisible: false,
                lastValueVisible: false,
                title: indicatorName,
            }, paneIndex); // Official paneIndex parameter

            const histogramData: HistogramData[] = validData.map(point => ({
                time: convertTime(point.time),
                value: point.value,
                color: point.value >= 0 ? this.theme.upColor : this.theme.downColor
            }));

            series.setData(histogramData);
            this.seriesMap.set(seriesKey, series);
        } else {
            // Add line series to specified pane using official API
            const series = this.chart.addSeries(LineSeries, {
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
            }, paneIndex); // Official paneIndex parameter

            const lineData: LineData[] = validData.map(point => ({
                time: convertTime(point.time),
                value: point.value
            }));

            series.setData(lineData);
            this.seriesMap.set(seriesKey, series);
        }

        // Set pane height using official PaneApi (fixed 150px)
        setTimeout(() => {
            const panes = this.chart.panes();
            if (panes[paneIndex]) {
                panes[paneIndex].setHeight(CHART_CONSTANTS.INDICATOR_PANE_HEIGHT);
            }
        }, 0);
    }

    private addReferenceLines(indicatorName: string, paneId: string, data: IndicatorDataPoint[]): void {
        const refLines = getReferenceLines(indicatorName);
        if (refLines.length === 0) return;

        const paneIndex = this.getPaneIndex(paneId);
        const timeRange = data.map(point => convertTime(point.time));
        const referenceLineSeries: ISeriesApi<SeriesType>[] = [];

        refLines.forEach((refLine) => {
            const series = this.chart.addSeries(LineSeries, {
                color: refLine.color,
                lineWidth: 1,
                lineStyle: refLine.style || 2, // Default to dashed
                priceLineVisible: false,
                lastValueVisible: false,
                title: refLine.label || `Reference ${refLine.value}`,
            }, paneIndex);

            const lineData = timeRange.map(time => ({ time, value: refLine.value }));
            series.setData(lineData);
            referenceLineSeries.push(series);
        });

        this.referenceLineMap.set(indicatorName, referenceLineSeries);
    }

    private removeReferenceLines(indicatorName: string): void {
        const lines = this.referenceLineMap.get(indicatorName);
        if (lines) {
            lines.forEach(line => this.chart.removeSeries(line));
            this.referenceLineMap.delete(indicatorName);
        }
    }

    private updateReferenceLines(indicatorName: string, data: IndicatorDataPoint[]): void {
        const lines = this.referenceLineMap.get(indicatorName);
        if (lines) {
            const timeRange = data.map(point => convertTime(point.time));
            const refLines = getReferenceLines(indicatorName);

            lines.forEach((line, index) => {
                if (refLines[index]) {
                    const lineData = timeRange.map(time => ({ time, value: refLines[index].value }));
                    (line as ISeriesApi<'Line'>).setData(lineData);
                }
            });
        }
    }

    private cleanupEmptyPanes(): void {
        // Remove panes that have no indicators
        const panesToRemove: string[] = [];
        
        for (const [paneId, config] of this.paneConfigurations) {
            if (paneId === 'main') continue; // Never remove main pane
            
            const hasIndicators = config.indicators.some(indicator => 
                this.seriesMap.has(`indicator_${indicator}`)
            );
            
            if (!hasIndicators) {
                panesToRemove.push(paneId);
            }
        }

        panesToRemove.forEach(paneId => {
            this.paneConfigurations.delete(paneId);
        });
    }

    private getPaneIndex(paneId: string): number {
        if (paneId === 'main') return 0;
        
        // For dynamic panes, find the index
        const dynamicPaneIds = Array.from(this.dynamicPanes.keys());
        const index = dynamicPaneIds.indexOf(paneId);
        
        if (index !== -1) {
            return index + 1; // +1 because main pane is at index 0
        }
        
        // Fallback for legacy panes
        const paneIds = Array.from(this.paneConfigurations.keys());
        const legacyIndex = paneIds.indexOf(paneId);
        return legacyIndex === -1 ? this.nextPaneIndex++ : legacyIndex;
    }

    private getIndicatorColor(indicatorName: string, colorIndex: number): string {
        const nameUpper = indicatorName.toUpperCase();
        
        // Special colors for specific indicators
        if (nameUpper.includes('RSI')) {
            return this.theme.rsiLineColor;
        } else if (nameUpper.includes('MACD')) {
            return '#2962FF'; // Blue for MACD
        } else if (nameUpper.includes('SIGNAL')) {
            return '#FF6D00'; // Orange for signal
        } else if (nameUpper.includes('HISTOGRAM')) {
            return '#FF7043'; // Red-orange for histogram
        }
        
        // Default color cycling
        return this.theme.indicatorColors[colorIndex % this.theme.indicatorColors.length];
    }

    private getColorIndex(indicatorName: string): number {
        // Simple hash function to get consistent color index
        let hash = 0;
        for (let i = 0; i < indicatorName.length; i++) {
            const char = indicatorName.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return Math.abs(hash);
    }

    private isHistogramIndicator(indicatorName: string): boolean {
        const nameUpper = indicatorName.toUpperCase();
        return nameUpper.includes('HISTOGRAM') || nameUpper.includes('MACD_HIST');
    }
}
