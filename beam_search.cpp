#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <algorithm>
#include <random>
#include <cmath>
#include <iomanip>

// ============================================================
// 数据结构定义
// ============================================================

// 一个候选序列
struct BeamCandidate {
    std::vector<int> tokens;       // 已生成的 token 序列
    double log_prob;               // 累积对数概率（越高越好）
    bool finished;                 // 是否已生成结束符

    BeamCandidate() : log_prob(0.0), finished(false) {}

    BeamCandidate(const std::vector<int>& t, double lp, bool fin = false)
        : tokens(t), log_prob(lp), finished(fin) {}
};

// 用于优先队列的比较（小顶堆，弹出概率最低的）
struct BeamComparator {
    bool operator()(const BeamCandidate& a, const BeamCandidate& b) const {
        return a.log_prob > b.log_prob; // 小顶堆
    }
};

// ============================================================
// 模拟的语言模型（实际中会是神经网络）
// ============================================================
class MockLanguageModel {
public:
    MockLanguageModel(int vocab_size, unsigned seed = 42)
        : vocab_size_(vocab_size), rng_(seed) {}

    // 给定当前序列，返回每个 token 的对数概率
    // 实际场景中这是神经网络的 softmax 输出
    std::vector<double> get_log_probs(const std::vector<int>& context) {
        std::vector<double> log_probs(vocab_size_);
        std::uniform_real_distribution<double> dist(-5.0, 0.0);

        for (int i = 0; i < vocab_size_; ++i) {
            log_probs[i] = dist(rng_);
        }

        // 模拟：让某些 token 更合理（基于上下文最后一个 token）
        if (!context.empty()) {
            int last = context.back();
            // 模拟 n-gram 偏好：某些 token 组合概率更高
            int preferred = (last + 1) % vocab_size_;
            log_probs[preferred] += 2.0;

            int second_choice = (last + 3) % vocab_size_;
            log_probs[second_choice] += 1.0;
        }

        // 归一化为 log softmax
        double max_val = *std::max_element(log_probs.begin(), log_probs.end());
        double sum = 0.0;
        for (auto& v : log_probs) {
            v = std::exp(v - max_val);
            sum += v;
        }
        for (auto& v : log_probs) {
            v = std::log(v / sum);
        }

        return log_probs;
    }

private:
    int vocab_size_;
    std::mt19937 rng_;
};

// ============================================================
// Beam Search 核心实现
// ============================================================
class BeamSearch {
public:
    BeamSearch(int beam_width, int max_length, int eos_token, double length_penalty = 0.0)
        : beam_width_(beam_width)
        , max_length_(max_length)
        , eos_token_(eos_token)
        , length_penalty_(length_penalty) {}

    // 执行 beam search
    std::vector<BeamCandidate> search(
        MockLanguageModel& model,
        const std::vector<int>& prefix = {})
    {
        // 初始化：从 prefix 开始
        std::vector<BeamCandidate> beams;
        beams.emplace_back(prefix, 0.0, false);

        std::cout << "=== Beam Search 开始 ===" << std::endl;
        std::cout << "Beam Width: " << beam_width_
                  << ", Max Length: " << max_length_
                  << ", Length Penalty: " << length_penalty_ << std::endl;
        std::cout << "Prefix: [";
        for (size_t i = 0; i < prefix.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << prefix[i];
        }
        std::cout << "]" << std::endl << std::endl;

        for (int step = 0; step < max_length_; ++step) {
            std::vector<BeamCandidate> all_candidates;

            // 对每个当前 beam 进行扩展
            for (const auto& beam : beams) {
                if (beam.finished) {
                    // 已完成的序列直接保留
                    all_candidates.push_back(beam);
                    continue;
                }

                // 获取下一个 token 的概率分布
                auto log_probs = model.get_log_probs(beam.tokens);

                // 为每个可能的 token 创建新候选
                for (int token = 0; token < (int)log_probs.size(); ++token) {
                    std::vector<int> new_tokens = beam.tokens;
                    new_tokens.push_back(token);

                    double new_log_prob = beam.log_prob + log_probs[token];
                    bool finished = (token == eos_token_);

                    all_candidates.emplace_back(new_tokens, new_log_prob, finished);
                }
            }

            // 按长度惩罚后的得分排序，保留 top-k
            std::sort(all_candidates.begin(), all_candidates.end(),
                [this](const BeamCandidate& a, const BeamCandidate& b) {
                    return score(a) > score(b);
                });

            // 保留 beam_width 个最优候选
            beams.clear();
            int active_count = 0;
            for (size_t i = 0; i < all_candidates.size() && beams.size() < (size_t)beam_width_; ++i) {
                beams.push_back(all_candidates[i]);
                if (!all_candidates[i].finished) active_count++;
            }

            // 打印当前步骤信息
            print_step(step + 1, beams);

            // 如果所有 beam 都已完成，提前终止
            if (active_count == 0) {
                std::cout << "\n所有候选序列已生成完毕，提前终止。" << std::endl;
                break;
            }
        }

        // 最终排序
        std::sort(beams.begin(), beams.end(),
            [this](const BeamCandidate& a, const BeamCandidate& b) {
                return score(a) > score(b);
            });

        return beams;
    }

private:
    // 长度惩罚得分：score = log_prob / (length ^ alpha)
    double score(const BeamCandidate& candidate) const {
        int length = std::max(1, (int)candidate.tokens.size());
        if (length_penalty_ > 0.0) {
            return candidate.log_prob / std::pow(length, length_penalty_);
        }
        return candidate.log_prob;
    }

    void print_step(int step, const std::vector<BeamCandidate>& beams) {
        std::cout << "Step " << step << ": ";
        for (size_t i = 0; i < beams.size(); ++i) {
            if (i > 0) std::cout << " | ";
            std::cout << "[";
            for (size_t j = 0; j < beams[i].tokens.size(); ++j) {
                if (j > 0) std::cout << ",";
                std::cout << beams[i].tokens[j];
            }
            std::cout << "] score=" << std::fixed << std::setprecision(4)
                      << score(beams[i]);
            if (beams[i].finished) std::cout << " ✓";
        }
        std::cout << std::endl;
    }

    int beam_width_;
    int max_length_;
    int eos_token_;
    double length_penalty_;
};

// ============================================================
// 主函数：演示
// ============================================================
int main() {
    // 参数设置
    const int VOCAB_SIZE = 10;       // 词表大小
    const int EOS_TOKEN = 0;         // 结束符
    const int BEAM_WIDTH = 3;        // 束宽
    const int MAX_LENGTH = 8;        // 最大生成长度
    const double LENGTH_PENALTY = 0.6; // 长度惩罚系数

    // 创建模型和搜索器
    MockLanguageModel model(VOCAB_SIZE, 42);
    BeamSearch searcher(BEAM_WIDTH, MAX_LENGTH, EOS_TOKEN, LENGTH_PENALTY);

    // 执行搜索
    auto results = searcher.search(model);

    // 输出最终结果
    std::cout << "\n=== 最终结果 ===" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "Rank " << (i + 1) << ": [";
        for (size_t j = 0; j < results[i].tokens.size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << results[i].tokens[j];
        }
        std::cout << "]" << std::endl;
        std::cout << "  Log Prob: " << std::fixed << std::setprecision(4)
                  << results[i].log_prob << std::endl;
        std::cout << "  Length:   " << results[i].tokens.size() << std::endl;
        std::cout << "  Score:    " << std::fixed << std::setprecision(4)
                  << results[i].log_prob / std::pow(
                      std::max(1, (int)results[i].tokens.size()), LENGTH_PENALTY)
                  << std::endl;
    }

    return 0;
}
