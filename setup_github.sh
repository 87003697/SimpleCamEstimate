#!/bin/bash

# GitHub仓库设置脚本
echo "🚀 V2M4相机搜索算法 - GitHub设置"
echo "=================================="

# 检查是否已经设置了远程仓库
if git remote get-url origin 2>/dev/null; then
    echo "✅ 远程仓库已设置:"
    git remote -v
else
    echo "📝 请按照以下步骤设置GitHub仓库:"
    echo ""
    echo "1. 在GitHub上创建新仓库:"
    echo "   - 仓库名建议: SimpleCamEstimate"
    echo "   - 描述: V2M4相机搜索算法简化版 - DUSt3R + PSO + 梯度下降"
    echo "   - 选择Public或Private"
    echo "   - 不要初始化README (我们已经有了)"
    echo ""
    echo "2. 创建后，复制仓库URL并运行:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/SimpleCamEstimate.git"
    echo ""
    echo "3. 推送代码:"
    echo "   git push -u origin main"
    echo ""
    echo "🔗 示例完整命令:"
    echo "git remote add origin https://github.com/zhiyuan-ma/SimpleCamEstimate.git"
    echo "git push -u origin main"
fi

echo ""
echo "📊 当前仓库状态:"
echo "   分支: $(git branch --show-current)"
echo "   提交数: $(git rev-list --count HEAD)"
echo "   文件数: $(git ls-files | wc -l)"

echo ""
echo "💡 提示:"
echo "   - 确保你已经登录GitHub账户"
echo "   - 如果使用HTTPS，可能需要Personal Access Token"
echo "   - 如果使用SSH，确保SSH密钥已配置" 