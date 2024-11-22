#pragma warning(disable : 4996) // 可删
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include<algorithm>
#include<queue>
#include<vector>
using namespace std;

#define iN 1288
#define iM 952
#define N (iN * iM)
#define M (N + iM) * 4
// iN 为重合图像最大高度，iM为宽度
// N为最大顶点数 M为最大边数
int n, m;      // AB图，n行m列
int A[iN][iM]; // A图
int B[iN][iM]; // B图
int C[iN][iM]; // 返回C图，左边为A，右边为B
int ecnt, t[M], nxt[M], head[M], val[M];
void init(int x)
{
    ecnt = 0;
    memset(head, 0, sizeof(head[0]) * x);
}
void addedge(int from, int to, int dis)
{
    t[++ecnt] = to;
    nxt[ecnt] = head[from];
    head[from] = ecnt;
    val[ecnt] = dis;
    //
    t[++ecnt] = from;
    nxt[ecnt] = head[to];
    head[to] = ecnt;
    val[ecnt] = dis;
}

void addedge2(int from, int to)
{
    t[++ecnt] = to;
    nxt[ecnt] = head[from];
    head[from] = ecnt;
    //
    t[++ecnt] = from;
    nxt[ecnt] = head[to];
    head[to] = ecnt;
}

int dis[N];
int pre[N];

int S, T;
#define maxquenum N
int que[maxquenum], inque[maxquenum]; // 循环队列；是否在队列里

typedef pair<int, int> pii;
void dijkstra(int start, int end)
{
    memset(pre, -1, sizeof(pre));
    memset(dis, 0x3f3f3f3f, sizeof dis);
    priority_queue<pii, vector<pii>, greater<pii>> q;
    q.push(make_pair(0, start)), dis[start] = 0;
    int* vis = inque;//
    while (!q.empty())
    {
        int u = q.top().second;
        q.pop();
        if (vis[u])
            continue;
        vis[u] = 1;
        if (u == T) {
            break;
        }
        for (int i = head[u]; i; i = nxt[i])
        {
            int v = t[i], w = val[i];
            if (!vis[v] && dis[v] > dis[u] + w)
            {
                dis[v] = dis[u] + w;
                pre[v] = u;
                q.push(make_pair(dis[v], v));
            }
        }
    }
}
/// 以下代码输出割线
typedef struct
{
    int u, v;
} po;
// po ret[N * 2];

int bfsgetC(int start)
{ // C为结果图
    for (int i = 0; i < n; i++)
    { // 把图B赋值给图C
        memcpy(C[i], B[i], sizeof(C[0][0]) * m);
    }
    memset(inque, 0, sizeof(inque[0]) * n * m);

    que[0] = start;
    inque[start] = 1;
    int x = start / m, y = start % m;

    C[x][y] = A[x][y];
    int qfront = 0, qtail = 1;
    int dir[4][2] = { 0, 1, 1, 0, 0, -1, -1, 0 };
    while (qfront < qtail) // 队列不空
    {
        int ele = que[qfront++];
        x = ele / m, y = ele % m;
        for (int i = 0; i < 4; i++)
        {
            int nx = x + dir[i][0], ny = dir[i][1] + y;
            int nele = nx * m + ny;
            if (!(nx >= 0 && nx < n && ny >= 0 && ny < m)) // 图像像素点不在(n,m)范围内
                continue;
            if (inque[nele])
                continue;
            int can = 1;
            for (int j = head[ele]; j; j = nxt[j])
            {
                int v = t[j];
                if (v == nele)
                { // 改边被cut过
                    can = 0;
                    break;
                }
            }
            if (can)
            {
                que[qtail++] = nele;
                inque[nele] = 1;
                C[nx][ny] = A[nx][ny];
            }
        }
    }
    return qtail;
}
// po ret[N];
void printl() // 输出图C
{
    int u = T, p = -1;
    int lenr = 0;
    po tmp;
    init(n * m);
    while (u != S)
    {
        p = pre[u];
        int realp = p;
        if (u == T)
        {
            // int w = abs(A[n - 1][j - 1] - B[n - 1][j - 1]) + abs(A[n - 1][j] - B[n - 1][j]);
            // int u = (n - 2) * (m - 1) + j;
            int j = p % (m - 1), i = n - 1;
            tmp.u = i * m + j - 1;
            tmp.v = tmp.u + 1;
            // ret[lenr++] = tmp;
        }
        else if (p == S)
        {
            tmp.u = u - 1;
            tmp.v = u;
            // ret[lenr++] = tmp;
        }
        else
        {
            if (abs(p - u) == 1) // 竖线
            {
                // int w = abs(A[i][j] - B[i][j]) + abs(A[i + 1][j] - B[i + 1][j]);
                // int u = i * (m - 1) + j;
                // int v = u + 1;
                if (p > u)
                {
                    // swap(p, u);
                    p = u;
                }
                int i = p / (m - 1), j = p % (m - 1);
                tmp.u = i * m + j;
                tmp.v = tmp.u + m;
                // ret[lenr++] = tmp;
            }
            else if (abs(u - p) == m - 1)
            {
                // int w = abs(A[i][j - 1] - B[i][j - 1]) + abs(A[i][j] - B[i][j]);
                // int u = (i - 1) * (m - 1) + j;
                // int v = u + (m - 1);
                if (u < p)
                {
                    p = u;
                }
                int i = p / (m - 1) + 1, j = p % (m - 1);
                tmp.u = i * m + j - 1;
                tmp.v = tmp.u + 1;
                // ret[lenr++] = tmp;
            }
            else
            {
                u = u;
            }
        }
        addedge2(tmp.u, tmp.v); // 记录u,v顶点对，后续bfs遍历不可访问u到v的化学键，因为uv将图像割开
        u = realp;
    }

    bfsgetC(0);
    puts("");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            printf("%03d ", C[i][j]);
        }
        puts("");
    }
}

// 主函数
int main()
{
    freopen("cut_crop.in", "r", stdin); // 打开文件，读取重合图像的数据
    scanf("%d%d", &n, &m);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            scanf("%d", &A[i][j]);
        }
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            scanf("%d", &B[i][j]);
        }
    }
    int start = 0, end = (n - 1) * (m - 1) + 1; // 连起点
    S = start;
    T = end;
    for (int j = 1; j < m; j++)
    {
        int w = abs(A[0][j - 1] - B[0][j - 1]) + abs(A[0][j] - B[0][j]);
        addedge(start, j, w);
    }
    // 横线
    for (int i = 1; i < n - 1; i++)
        for (int j = 1; j < m; j++)
        {
            int w = abs(A[i][j - 1] - B[i][j - 1]) + abs(A[i][j] - B[i][j]);
            int u = (i - 1) * (m - 1) + j;
            int v = u + (m - 1);
            addedge(u, v, w);
        }
    // 连终点
    for (int j = 1; j < m; j++)
    {
        int w = abs(A[n - 1][j - 1] - B[n - 1][j - 1]) + abs(A[n - 1][j] - B[n - 1][j]);
        int u = (n - 2) * (m - 1) + j;
        addedge(u, end, w);
    }
    //
    for (int i = 0; i < n - 1; i++) // 竖线
        for (int j = 1; j < m - 1; j++)
        {
            int w = abs(A[i][j] - B[i][j]) + abs(A[i + 1][j] - B[i + 1][j]);
            int u = i * (m - 1) + j;
            int v = u + 1;
            addedge(u, v, w);
        }
    clock_t tbegin = clock();
    dijkstra(start, end);
    clock_t tend = clock();
    double time_consumption = (double)(tend - tbegin) / CLOCKS_PER_SEC;

    printf("dij timecost %f", time_consumption);

    freopen("cutdij0617.txt", "w", stdout);
    printf("%d\n", dis[end]);
    printl();
    return 0;
}