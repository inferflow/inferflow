#pragma once

#include "naive_matrix.h"
#include "std_static_graph.h"
#include "chained_str_stream.h"

namespace sslib
{

class GraphHelper
{
public:
    struct MessageSpec
    {
        string graph_name;
        bool show_info = true;
        bool show_error = true;

        MessageSpec(const string &p_graph_name = "", bool p_show_info = true, bool p_show_error = true)
        {
            graph_name = p_graph_name;
            show_info = p_show_info;
            show_error = p_show_error;
        }
    };

public:
    template <class WeightType>
    static bool LoadGraph(StdStaticGraph<WeightType> &graph, const string &path,
        const MessageSpec &spec, bool sort_by_weight = true, uint32_t max_loaded_nbrs = UINT32_MAX)
    {
        bool ret = true;
        if (spec.show_info) {
            LogKeyInfo("Loading the %s graph...", spec.graph_name.c_str());
        }
        if (max_loaded_nbrs == UINT32_MAX) {
            ret = graph.Load(path, "");
        }
        else {
            ret = graph.LoadFmtCompactBin(path, max_loaded_nbrs);
        }

        if (!ret)
        {
            if (spec.show_error) {
                LogError("Failed to load the graph from %s", path.c_str());
            }
            return false;
        }

        if (sort_by_weight) {
            graph.Sort(GraphConst::SortByWeight, false);
        }
        else {
            graph.Sort(GraphConst::SortByIdx, true);
        }

        return ret;
    }

    template <class WeightType>
    static bool LoadGraph(StdStaticGraph<WeightType> &graph, const string &path,
        const string &graph_name, bool sort_by_weight = true, uint32_t max_loaded_nbrs = UINT32_MAX)
    {
        MessageSpec spec(graph_name);
        return LoadGraph(graph, path, spec, sort_by_weight, max_loaded_nbrs);
    }

    template <class WeightType>
    static bool LoadGraph(StdStaticGraph<WeightType> &graph, IBinStream &reader,
        const MessageSpec &spec, bool sort_by_weight = true, uint32_t max_loaded_nbrs = UINT32_MAX)
    {
        bool ret = true;
        if (spec.show_info) {
            LogKeyInfo("Loading the %s graph...", spec.graph_name.c_str());
        }
        ret = graph.LoadFmtCompactBin(reader, max_loaded_nbrs);
        if (!ret)
        {
            if (spec.show_error) {
                LogError("Failed to load the graph from the stream");
            }
            return false;
        }

        if (sort_by_weight) {
            graph.Sort(GraphConst::SortByWeight, false);
        }
        else {
            graph.Sort(GraphConst::SortByIdx, true);
        }

        return ret;
    }

    template <class WeightType>
    static bool LoadGraph(StdStaticGraph<WeightType> &graph, IBinStream &reader,
        const string &graph_name, bool sort_by_weight = true, uint32_t max_loaded_nbrs = UINT32_MAX)
    {
        MessageSpec spec(graph_name);
        return LoadGraph(graph, reader, spec, sort_by_weight, max_loaded_nbrs);
    }

    template <class WeightType>
    static bool LoadGraph(NaiveMatrix<WeightType> &graph, const string &path, const string &graph_name)
    {
        LogKeyInfo("Loading the %s graph...", graph_name.c_str());
        bool ret = graph.LoadFmtCompactBin(path);
        if (!ret) {
            LogError("Failed to load the graph from %s", path.c_str());
            return false;
        }

        return ret;
    }

    template <class WeightType>
    static bool SaveGraph(const NaiveMatrix<WeightType> &graph, const string &path,
        const MessageSpec &spec, bool is_reverse = false, bool is_order_by_weight = true)
    {
        if (spec.show_info) {
            LogKeyInfo("Saving the %s graph...", spec.graph_name.c_str());
        }
        bool ret = graph.SaveFmtCompactBin(path, UINT32_MAX, is_reverse, is_order_by_weight);
        if (!ret)
        {
            if (spec.show_error) {
                LogError("Failed to save the %s graph to %s", spec.graph_name.c_str(), path.c_str());
            }
            return false;
        }

        return ret;
    }

    template <class WeightType>
    static bool SaveGraph(const NaiveMatrix<WeightType> &graph, const string &path,
        const string &graph_name, bool is_reverse = false, bool is_order_by_weight = true)
    {
        MessageSpec spec(graph_name);
        return SaveGraph(graph, path, spec, is_reverse, is_order_by_weight);
    }

    template <class WeightType>
    static bool SaveGraph(const NaiveMatrix<WeightType> &graph, IBinStream &writer,
        const MessageSpec &spec, bool is_reverse = false, bool is_order_by_weight = true)
    {
        if (spec.show_info) {
            LogKeyInfo("Saving the %s graph...", spec.graph_name.c_str());
        }
        bool ret = graph.SaveFmtCompactBin(writer, UINT32_MAX, is_reverse, is_order_by_weight);
        if (!ret)
        {
            if (spec.show_error) {
                LogError("Failed to save the %s graph to the stream", spec.graph_name.c_str());
            }
            return false;
        }

        return ret;
    }

    template <class WeightType>
    static bool SaveGraph(const NaiveMatrix<WeightType> &graph, IBinStream &writer,
        const string &graph_name, bool is_reverse = false, bool is_order_by_weight = true)
    {
        MessageSpec spec(graph_name);
        return SaveGraph(graph, writer, spec, is_reverse, is_order_by_weight);
    }

    template <class WeightType>
    static bool SaveGraph(const StdStaticGraph<WeightType> &graph, const string &path,
        const string &graph_name)
    {
        LogKeyInfo("Saving the %s graph...", graph_name.c_str());
        bool ret = graph.StoreFmtCompactBin(path);
        if (!ret) {
            LogError("Failed to save the %s graph to %s", graph_name.c_str(), path.c_str());
            return false;
        }

        return ret;
    }

    template <class WeightType>
    static bool SaveGraph(const StdStaticGraph<WeightType> &graph, IBinStream &writer,
        const string &graph_name)
    {
        LogKeyInfo("Saving the %s graph...", graph_name.c_str());
        bool ret = graph.StoreFmtCompactBin(writer);
        if (!ret) {
            LogError("Failed to save the %s graph to the stream", graph_name.c_str());
            return false;
        }

        return ret;
    }

    template <class WeightType>
    static bool TransGraph(StdStaticGraph<WeightType> &target, const NaiveMatrix<WeightType> &source,
        bool sort_by_weight = true, bool be_reverse = false)
    {
        GraphHelper::MessageSpec spec;
        spec.show_error = false;
        spec.show_info = false;

        ChainedStrStream strm;
        bool ret = GraphHelper::SaveGraph(source, strm, spec, be_reverse);
        ret = ret && GraphHelper::LoadGraph(target, strm, spec, sort_by_weight);
        return ret;
    }
};

} //end of namespace
