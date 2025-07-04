/**
 * Series components for TradingView Lightweight Charts
 */

import React, { 
    forwardRef, 
    useImperativeHandle, 
    useLayoutEffect, 
    useRef 
} from 'react';
import { 
    ISeriesApi, 
    CandlestickSeries, 
    LineSeries, 
    HistogramSeries,
    CandlestickData,
    LineData,
    HistogramData,
    SeriesMarker,
    Time
} from 'lightweight-charts';
import { useChart } from './hooks';
import { OHLCVDataPoint, IndicatorDataPoint, convertTime } from './types';

export interface CandlestickSeriesProps {
    data: OHLCVDataPoint[];
    markers?: SeriesMarker<Time>[];
    upColor?: string;
    downColor?: string;
    wickUpColor?: string;
    wickDownColor?: string;
    title?: string;
    paneIndex?: number;
}

export interface LineSeriesProps {
    data: IndicatorDataPoint[];
    color?: string;
    lineWidth?: number;
    title?: string;
    paneIndex?: number;
}

export interface HistogramSeriesProps {
    data: IndicatorDataPoint[];
    color?: string;
    title?: string;
    paneIndex?: number;
}

// Candlestick Series Component
export const CandlestickSeriesComponent = forwardRef<ISeriesApi<'Candlestick'>, CandlestickSeriesProps>(
    (props, ref) => {
        const parent = useChart();
        const context = useRef<{
            _api?: ISeriesApi<'Candlestick'>;
            api(): ISeriesApi<'Candlestick'>;
            free(): void;
        }>({
            api() {
                if (!this._api) {
                    const { data, markers, ...seriesOptions } = props;
                    
                    this._api = parent.api().addSeries(CandlestickSeries, {
                        upColor: '#26a69a',
                        downColor: '#ef5350',
                        borderVisible: false,
                        wickUpColor: '#26a69a',
                        wickDownColor: '#ef5350',
                        priceLineVisible: false,
                        lastValueVisible: true,
                        title: 'Price',
                        ...seriesOptions,
                    });
                    
                    // Convert and set data
                    const candleData: CandlestickData[] = data.map(d => ({
                        time: convertTime(d.time),
                        open: d.open,
                        high: d.high,
                        low: d.low,
                        close: d.close,
                    }));
                    
                    this._api.setData(candleData);
                    
                    // Set markers if provided
                    if (markers && markers.length > 0) {
                        this._api.setMarkers(markers);
                    }
                }
                return this._api;
            },
            
            free() {
                if (this._api && !parent.isRemoved) {
                    parent.free(this._api);
                }
            },
        });

        useLayoutEffect(() => {
            const currentRef = context.current;
            currentRef.api();

            return () => currentRef.free();
        }, []);

        // Update data when props change
        useLayoutEffect(() => {
            const currentRef = context.current;
            if (currentRef._api && props.data) {
                const candleData: CandlestickData[] = props.data.map(d => ({
                    time: convertTime(d.time),
                    open: d.open,
                    high: d.high,
                    low: d.low,
                    close: d.close,
                }));
                
                currentRef._api.setData(candleData);
                
                // Update markers
                if (props.markers && props.markers.length > 0) {
                    currentRef._api.setMarkers(props.markers);
                }
            }
        }, [props.data, props.markers]);

        // Update options when props change
        useLayoutEffect(() => {
            const currentRef = context.current;
            if (currentRef._api) {
                const { data, markers, ...seriesOptions } = props;
                currentRef._api.applyOptions(seriesOptions);
            }
        }, [props.upColor, props.downColor, props.wickUpColor, props.wickDownColor, props.title]);

        useImperativeHandle(ref, () => context.current.api(), []);

        return null;
    }
);

CandlestickSeriesComponent.displayName = 'CandlestickSeries';

// Line Series Component
export const LineSeriesComponent = forwardRef<ISeriesApi<'Line'>, LineSeriesProps>(
    (props, ref) => {
        const parent = useChart();
        const context = useRef<{
            _api?: ISeriesApi<'Line'>;
            api(): ISeriesApi<'Line'>;
            free(): void;
        }>({
            api() {
                if (!this._api) {
                    const { data, ...seriesOptions } = props;
                    
                    // Add to specific pane if specified
                    const addOptions = {
                        color: '#2196F3',
                        lineWidth: 2,
                        priceFormat: {
                            type: 'price' as const,
                            precision: 4,
                            minMove: 0.0001,
                        },
                        priceLineVisible: false,
                        lastValueVisible: false,
                        ...seriesOptions,
                    };
                    
                    if (props.paneIndex !== undefined && props.paneIndex > 0) {
                        // Add to specific pane
                        this._api = parent.api().addSeries(LineSeries, addOptions, props.paneIndex);
                    } else {
                        // Add to main chart (pane 0)
                        this._api = parent.api().addSeries(LineSeries, addOptions);
                    }
                    
                    // Convert and set data
                    const lineData: LineData[] = data
                        .filter(d => d.value != null && !isNaN(d.value))
                        .map(d => ({
                            time: convertTime(d.time),
                            value: d.value,
                        }));
                    
                    this._api.setData(lineData);
                }
                return this._api;
            },
            
            free() {
                if (this._api && !parent.isRemoved) {
                    parent.free(this._api);
                }
            },
        });

        useLayoutEffect(() => {
            const currentRef = context.current;
            currentRef.api();

            return () => currentRef.free();
        }, []);

        // Update data when props change
        useLayoutEffect(() => {
            const currentRef = context.current;
            if (currentRef._api && props.data) {
                const lineData: LineData[] = props.data
                    .filter(d => d.value != null && !isNaN(d.value))
                    .map(d => ({
                        time: convertTime(d.time),
                        value: d.value,
                    }));
                
                currentRef._api.setData(lineData);
            }
        }, [props.data]);

        // Update options when props change
        useLayoutEffect(() => {
            const currentRef = context.current;
            if (currentRef._api) {
                const { data, paneIndex, ...seriesOptions } = props;
                currentRef._api.applyOptions(seriesOptions);
            }
        }, [props.color, props.lineWidth, props.title]);

        useImperativeHandle(ref, () => context.current.api(), []);

        return null;
    }
);

LineSeriesComponent.displayName = 'LineSeries';

// Histogram Series Component
export const HistogramSeriesComponent = forwardRef<ISeriesApi<'Histogram'>, HistogramSeriesProps>(
    (props, ref) => {
        const parent = useChart();
        const context = useRef<{
            _api?: ISeriesApi<'Histogram'>;
            api(): ISeriesApi<'Histogram'>;
            free(): void;
        }>({
            api() {
                if (!this._api) {
                    const { data, ...seriesOptions } = props;
                    
                    const addOptions = {
                        priceFormat: { type: 'volume' as const },
                        priceScaleId: 'volume_scale',
                        lastValueVisible: false,
                        priceLineVisible: false,
                        ...seriesOptions,
                    };
                    
                    if (props.paneIndex !== undefined && props.paneIndex > 0) {
                        // Add to specific pane
                        this._api = parent.api().addSeries(HistogramSeries, addOptions, props.paneIndex);
                    } else {
                        // Add to main chart (pane 0)
                        this._api = parent.api().addSeries(HistogramSeries, addOptions);
                        
                        // Configure volume scale for main chart
                        parent.api().priceScale('volume_scale').applyOptions({
                            scaleMargins: { top: 0.7, bottom: 0 },
                        });
                    }
                    
                    // Convert and set data
                    const histogramData: HistogramData[] = data
                        .filter(d => d.value != null && !isNaN(d.value))
                        .map(d => ({
                            time: convertTime(d.time),
                            value: d.value,
                            color: d.value >= 0 ? '#26a69a' : '#ef5350',
                        }));
                    
                    this._api.setData(histogramData);
                }
                return this._api;
            },
            
            free() {
                if (this._api && !parent.isRemoved) {
                    parent.free(this._api);
                }
            },
        });

        useLayoutEffect(() => {
            const currentRef = context.current;
            currentRef.api();

            return () => currentRef.free();
        }, []);

        // Update data when props change
        useLayoutEffect(() => {
            const currentRef = context.current;
            if (currentRef._api && props.data) {
                const histogramData: HistogramData[] = props.data
                    .filter(d => d.value != null && !isNaN(d.value))
                    .map(d => ({
                        time: convertTime(d.time),
                        value: d.value,
                        color: d.value >= 0 ? '#26a69a' : '#ef5350',
                    }));
                
                currentRef._api.setData(histogramData);
            }
        }, [props.data]);

        // Update options when props change
        useLayoutEffect(() => {
            const currentRef = context.current;
            if (currentRef._api) {
                const { data, paneIndex, ...seriesOptions } = props;
                currentRef._api.applyOptions(seriesOptions);
            }
        }, [props.color, props.title]);

        useImperativeHandle(ref, () => context.current.api(), []);

        return null;
    }
);

HistogramSeriesComponent.displayName = 'HistogramSeries';
