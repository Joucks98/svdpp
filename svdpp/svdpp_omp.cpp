#define _USE_MATH_DEFINES
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <math.h>
#include <fstream>
#include <numeric>
#include <map>
#include <tuple>
#include <algorithm>
#include <random>
#include <set>
#include <functional>
#include <iomanip>
#include <assert.h>
#include <iterator>
#include <exception>
#include <omp.h>
#include "cmdline.h"

using namespace std;
using std::default_random_engine;
using std::uniform_int_distribution;
using std::cout;

using namespace Eigen;

typedef Eigen::SparseMatrix<double, StorageOptions::RowMajor> SpMat;// declares a column-major sparse matrix type of double
typedef SpMat SpMat_csr;
typedef Eigen::Triplet<double> T;
typedef Eigen::SparseVector<double> SpVec;

const double reserve_ratio = 1.0 / 3;

VectorXd zY(const vector<int>& ujList, const MatrixXd& Y)
{
    int n = (int)Y.cols();
    SpVec tmp(n);
    tmp.reserve(static_cast<int>(n* reserve_ratio));
    for (auto& j : ujList)
    {
        tmp.coeffRef(j) = 1.0;
    }
    VectorXd z = Y*tmp;
    double ru = 0.0;
    if (!ujList.empty())
    {
        ru = 1.0 / sqrt(ujList.size());
        z *= ru; 
    }
    assert(Eigen::isinf(z.array()).any() == 0);
    return z;
}

double predict_ui(const RowVectorXd& Pu, const VectorXd& Qi, const RowVectorXd& z, double bu, double bi, double mu)
{
    return (Pu + z)*Qi + bu + bi + mu;
}

MatrixXd predict(const MatrixXd& P, const MatrixXd& Q, const MatrixXd& Y, const SpMat& Z
    , const VectorXd& bu, const VectorXd& bi, double mu)
{
    MatrixXd tmp = Y.transpose(); // 74ms
    tmp = Z * tmp; //525s
    tmp += P;
    tmp *= Q;//240s
    tmp += bu * RowVectorXd::Ones(bi.size());//90s
    tmp += VectorXd::Ones(bu.size())*(bi.transpose());//120s
    return tmp.array() + mu;
}

double costFunc(const map<int, pair<vector<int>, vector<double>>>& uirMap,
    const MatrixXd& P, const MatrixXd& Q, const VectorXd& bu, const VectorXd& bi, double mu, const MatrixXd& Y)
{
    int m = (int)P.rows();
    int n = (int)Q.cols();
    int F = (int)Q.rows();
    vector<double> costVec(m, 0.0);

#pragma omp parallel for // 23s
    for (int u = 0; u < m; ++u)
    {
        VectorXd z = VectorXd::Zero(F);
        auto itr = uirMap.find(u);
        if (itr != uirMap.end())
        {
            const auto& ujList = (*itr).second.first;
            z = zY(ujList, Y);
            vector<double> uPredictErrorVec((*itr).second.second);
#pragma omp parallel for
            for (int k = 0; k < (int)ujList.size(); ++k)
            {
                uPredictErrorVec[k] -= predict_ui(P.row(u), Q.col(ujList[k]), z.transpose(), bu(u), bi(ujList[k]), mu);
            }
            costVec[u] = accumulate(uPredictErrorVec.begin(), uPredictErrorVec.end(), 0.0, [](double re, double e) { 
                return re + e * e; 
            });
        }        
    }
    double cst = 0.0;
#pragma omp parallel for reduction(+:cst) // 33ms
    for (int u = 0; u < m; ++u)
    {
        cst += costVec[u];
    }
    return cst;
}

double costFunc(const vector<T>& data, const map<int, pair<vector<int>, vector<double>>>& uirMap,
    const MatrixXd& P, const MatrixXd& Q, const VectorXd& bu, const VectorXd& bi, double mu, const MatrixXd& Y)
{
    int numT = (int)(data.size());
    vector<double> diffVec(numT, 0.0);
    int m = (int)P.rows();
    int n = (int)Q.cols();
    int F = (int)Q.rows();

#pragma omp parallel for
    for (int k = 0; k < numT; ++k)
    {
        int u = data[k].row();
        int i = data[k].col();
        if (u >= m || i >= n)
        {
            continue;
        }
        VectorXd z = VectorXd::Zero(F);
        auto itr = uirMap.find(u);
        if (itr != uirMap.end())
        {
            z = zY((*itr).second.first, Y);
        }
        double r = data[k].value() - predict_ui(P.row(u), Q.col(i), z.transpose(), bu(u), bi(i), mu);
        diffVec[k] = r * r;
    }
    double cst = 0.0;
#pragma omp parallel for reduction(+:cst)
    for (int k = 0; k < numT; ++k)
    {
        cst += diffVec[k];
    }
    return cst;
}


map<int, pair<vector<int>,vector<double>>> userMapItemValueListFromMatrix(const SpMat_csr& mat)
{
    map<int, pair<vector<int>, vector<double>>> tmp;
    for (int u = 0; u < (int)mat.rows(); ++u)
    {
        int unnz = mat.outerIndexPtr()[u + 1] - mat.outerIndexPtr()[u];
        if (unnz == 0)
        {
            continue;
        }
        auto sptr = &mat.innerIndexPtr()[mat.outerIndexPtr()[u]];
        vector<int> ujList(sptr, sptr+unnz);
        vector<double> uValueList(unnz);
        //generate(uValueList.begin(), uValueList.end(), [i = 0, u, &sptr, &mat]()mutable { return mat.coeff(u, sptr[i++]); });
        transform(ujList.begin(), ujList.end(), uValueList.begin(), [&mat, u](int& j) { return mat.coeff(u, j); });
        tmp[u] = make_pair(std::move(ujList), std::move(uValueList));
    }
    return tmp;
}

vector<T> ijvTuplesFromMatrix(const SpMat_csr& mat)
{
    vector<T> tmp;
    tmp.reserve(mat.nonZeros());
    for (int k = 0; k < mat.outerSize(); ++k)
    {
        for (SpMat::InnerIterator it(mat, k); it; ++it)
        {
            tmp.emplace_back(it.row(), it.col(), it.value());
        }
    }
    return tmp;
}

void gduData(int u, const vector<int>& ujList, const vector<double>& uValueList, double alpha, double lambda, double mu,
    MatrixXd *P, MatrixXd *Q, MatrixXd* Y, VectorXd* bu, VectorXd* bi)
{
    //vector<int> ujList(ujValueList.size());
    //transform(ujValueList.begin(), ujValueList.end(), ujList.begin(), [](pair<int, double>& a) { return a.first; });

    //double ru = 0.0;
    assert(!ujList.empty());
    VectorXd z = zY(ujList, *Y);
    vector<double> eujList(ujList.size(), 0.0);
    VectorXd puz = (*P).row(u).transpose() + z;
    int numUj = (int)(ujList.size());
#pragma omp parallel for
    for (int k = 0; k < numUj; ++k)
    {
        eujList[k] = uValueList[k] - ((*Q).col(ujList[k]).dot(puz) + (*bu)(u) + (*bi)(ujList[k]) + mu);
    }

    double eujListSum = 0.0;
#pragma omp parallel for reduction(+:eujListSum)
    for (int k = 0; k < numUj; ++k)
    {
        eujListSum += eujList[k];
    }

    (*bu)(u) += alpha * (eujListSum - lambda * (*bu)(u));
#pragma omp parallel for
    for (int k = 0; k < numUj; ++k)
    {
        (*bi)(ujList[k]) += alpha * (eujList[k] - lambda * (*bi)(ujList[k]));
    }

    SpVec tmp((*bi).size());
    tmp.reserve(static_cast<int>((*bi).size()*reserve_ratio));
    for (int k = 0; k < numUj; ++k) // 这里不能并行化
    {
        tmp.coeffRef(ujList[k]) = eujList[k];
    }
    VectorXd Qeuj = (*Q) * tmp;
    RowVectorXd dPu = Qeuj.transpose() - lambda * (*P).row(u);
    (*P).row(u) += alpha * dPu;

    double ru = 1.0 / sqrt(ujList.size());
    VectorXd temp = ru * Qeuj;
#pragma omp parallel for
    for (int k = 0; k < numUj; ++k)
    {
        VectorXd dQuj = puz * eujList[k] - lambda * (*Q).col(ujList[k]);
        VectorXd dYuj = temp - lambda * (*Y).col(ujList[k]);

        (*Q).col(ujList[k]) += alpha * dQuj;
        (*Y).col(ujList[k]) += alpha * dYuj;
    }
}

double metropolis(const vector<int>& uSeq, const map<int, pair<vector<int>, vector<double>>>& uiMap, double alpha, double lambda, double mu, double initCost,
    MatrixXd *P, MatrixXd *Q, MatrixXd* Y, VectorXd* bu, VectorXd* bi)
{
    MatrixXd pre_P = *P, pre_Q = *Q, pre_Y = *Y;
    VectorXd pre_bu = *bu, pre_bi = *bi;
    const double CC = 1e8;
    default_random_engine e(3);
    uniform_real_distribution<double> ud(0.0, 1.0);
    double pre_cost = initCost, cur_cost = initCost;
    for (auto& u : uSeq)
    {
        auto itr = uiMap.find(u);
        /*if (itr == uiMap.end())
        {
            continue;
        }*/
        gduData(u, (*itr).second.first, (*itr).second.second, alpha, lambda, mu, P, Q, Y, bu, bi);
        /*double cur_cost = costFunc(uiMap, *P, *Q, *bu, *bi, mu, *Y);
        if (cur_cost >= pre_cost)
        {
            double rng = exp(-(cur_cost - pre_cost) / (alpha * CC));
            if (rng <= ud(e))
            {
                *P = pre_P;
                *Q = pre_Q;
                *Y = pre_Y;
                *bu = pre_bu;
                *bi = pre_bi;
                continue;
            }            
        }
        pre_P = *P;
        pre_Q = *Q;
        pre_Y = *Y;
        pre_bu = *bu;
        pre_bi = *bi;
        pre_cost = cur_cost;*/
    }
    return costFunc(uiMap, *P, *Q, *bu, *bi, mu, *Y)/*pre_cost*/;
}


SpMat constructZ1(SpMat& mat)
{
    bool c = mat.isCompressed();
    if (c)
    {
        mat.uncompress();
    }
    SpMat Z = mat;
    vector<int> nVec(mat.innerNonZeroPtr(), mat.innerNonZeroPtr() + mat.outerSize());
    //int ns = accumulate(nVec.begin(), nVec.end(), 0);
    vector<double> ruVec(nVec.size());
    transform(nVec.begin(), nVec.end(), ruVec.begin(), [](int nu) { return nu ? 1.0 / sqrt(nu) : 0; });
    vector<double> vv(Z.nonZeros());
    auto itr = vv.begin();
    auto vItr = ruVec.begin();
    for (auto n : nVec)
    {
        while (n--)
        {
            *itr++ = *vItr;
        }
        ++vItr;
    }
#pragma omp parallel for
    for (int k = 0; k < (int)Z.nonZeros(); ++k)
    {
        Z.data().value(k) = vv[k];
    }
    Z.makeCompressed();
    
    if (c)
    {
        mat.makeCompressed();
    }
    return Z;
}

void learningSVDpp(SpMat_csr& train_csr, const vector<T>& validationSet, MatrixXd *P, MatrixXd *Q, VectorXd* bu, VectorXd* bi, double mu, MatrixXd* Y
    , int nEpoches, std::function<double(int)> stepFunc, double lambda)
{
    int m = (int)train_csr.rows();
    int n = (int)train_csr.cols();
    int F = (int)(*Q).rows();

    auto uirMap = userMapItemValueListFromMatrix(train_csr);
    

    cout << fixed << setprecision(5);
    cout << setfill(' ') << right;
    cout << "start iterations ..." << endl;

    vector<int> uVec(uirMap.size());
    std::transform(uirMap.begin(), uirMap.end(), uVec.begin(), [](decltype(*uirMap.begin())& a) { return a.first; });

    int bestEpoch = 0;
    double best_validation_rmse = 1e9;
    int delayCount = 0;
    MatrixXd re_P, re_Q, re_Y;
    VectorXd re_bu, re_bi;
    
    double preCost1 = costFunc(uirMap, *P, *Q, *bu, *bi, mu, *Y);
    for (int epoch = 0; epoch < nEpoches; ++epoch)
    {
        double alpha = stepFunc(epoch);
        shuffle(uVec.begin(), uVec.end(), std::default_random_engine(epoch));
        preCost1 = metropolis(uVec, uirMap, alpha, lambda, mu, preCost1, P, Q, Y, bu, bi);
        double cost1 = preCost1;
        double cost2 = costFunc(validationSet, uirMap, *P, *Q, *bu, *bi, mu, *Y);
        double penalty = lambda * ((*bu).squaredNorm() + (*bi).squaredNorm() + \
            (*P).squaredNorm() + (*Q).squaredNorm() + (*Y).squaredNorm());

        double train_rmse = sqrt(cost1 / train_csr.nonZeros());
        double validation_rmse = sqrt(cost2 / validationSet.size());
        cout << "Iteration:" << setw(4) << epoch \
            << ",    rmse:" << setw(12) << setprecision(9) << train_rmse\
            << ",    validation rmse:" << setw(12) << validation_rmse\
            << ",    cost:" << setprecision(5) << setw(12) << cost1 \
            << ",    penalty:" << setw(12) << penalty << endl;
        if (validation_rmse < best_validation_rmse)
        {
            bestEpoch = epoch;
            best_validation_rmse = validation_rmse;
            delayCount = 0;
            re_P = *P;
            re_Q = *Q;
            re_Y = *Y;
            re_bu = *bu;
            re_bi = *bi;            
        }
        else
        {
            if (++delayCount == 10)
            {
                break;
            }
        }
    }
    cout.unsetf(ios::fixed);
    cout << "best epoch :" << bestEpoch << endl;

    *P = re_P;
    *Q = re_Q;
    *Y = re_Y;
    *bu = re_bu;
    *bi = re_bi;
}

void initParams(MatrixXd* P, MatrixXd* Q, VectorXd* bu, VectorXd* bi, MatrixXd* Y, int m, int n, int F)
{
    srand(1/*(unsigned)time(NULL)*/);

    P->setRandom(m, F);
    P->array() = (1 + P->array())*.5 / sqrt(F);
    Q->setRandom(F, n);
    Q->array() = (1 + Q->array())*.5 / sqrt(F);
    Y->setRandom(F, n);

    bu->setZero(m);
    bi->setZero(n);
}

void setNegSample(SpMat& train_csr, int ratio = 1)
{
    train_csr.reserve(VectorXi::Constant(train_csr.rows(), 1000));

    srand(3);
    MatrixXd randM = MatrixXd::Random(train_csr.rows(), train_csr.cols());//74s
    randM.array() += 1;//39s
    randM.array() *= 0.5;//39s
    double threshold = (1.0*ratio* train_csr.nonZeros()) / train_csr.size();
#pragma omp parallel for //40s
    for (int i = 0; i < (int)(train_csr.rows()); ++i)
    {
#pragma omp parallel for
        for (int j = 0; j < (int)train_csr.cols(); ++j)
        {
            if (randM(i,j) < threshold && (train_csr.coeff(i, j) == 0))
            {
                train_csr.coeffRef(i, j) = 1e-10;
            }
        }
    }

    train_csr.makeCompressed();
}

pair<vector<double>, vector<double>> precisionRecall(const vector<T> & indexedTestData, const MatrixXi& recIndexMat)
{
    map<int, set<int>> testUiMap;
    for (const T& t : indexedTestData)
    {
        int u = t.row();
        auto it = testUiMap.emplace(u, set<int>());
        it.first->second.insert(t.col());
    }


    int N = (int)recIndexMat.cols();
    vector<double> numerator(N+1, 0.0);
    for (int i = 0; i < N; ++i)
    {
        VectorXi tmp = recIndexMat.col(i);

        for (auto& tui: testUiMap)
        {
            int u = tui.first;
            auto& iSet = tui.second;
            if (iSet.find(tmp(u)) != iSet.end())
            {
                numerator[i] += 1;
            }
        }        
    }
    numerator.resize(N);

    vector<double> precisionVec(N);
    int intersectUserNum = (int)testUiMap.size();
    std::transform(numerator.begin(), numerator.end(), precisionVec.begin(), [intersectUserNum, i = 1](double a) mutable {
        return a / (intersectUserNum*i++);
    });

    vector<double> recallVec(N);
    double denominator = accumulate(testUiMap.begin(), testUiMap.end(), 0.0, [](double re, pair<const int, set<int>>& a) {
        return re + a.second.size();
    });
    std::transform(numerator.begin(), numerator.end(), recallVec.begin(), [denominator](double v) { return v / denominator; });

    //Matrix2Xd prMat(2, N);
    return make_pair(std::move(precisionVec), std::move(recallVec));
}

Matrix<int,Dynamic,Dynamic,RowMajor>
recommendIndexMatrix(const MatrixXd& rateMat, int N)
{
    int nr = static_cast<int>(rateMat.rows());
    int nc = static_cast<int>(rateMat.cols());
    Matrix<int, Dynamic, Dynamic, RowMajor> indMat;
    indMat.setOnes(nr, nc);
    DiagonalMatrix<int, Dynamic> indDiagMat(nc);
    indDiagMat.diagonal() = VectorXi::LinSpaced(nc, 0, nc - 1);
    indMat *= indDiagMat;

#pragma omp parallel for
    for (int u = 0; u < nr; ++u)
    {
        auto pStart = indMat.data() + u * nc;
        auto pMid = pStart + N;
        std::partial_sort(pStart, pMid, pStart + nc, [&rateMat, u](int i, int j) {
            return rateMat(u, i) > rateMat(u, j);
        });
    }
    cout << "sort rateMat finished." << endl;

    return indMat.leftCols(N);
}

map<int, vector<int>> recommend(const MatrixXi& recIdxMat, const vector<int>& idUserMapList, const vector<int>& idItemMapList)
{
    
    /*map<int, pair<vector<int>,vector<double>>> re;
    for (int i = 0; i < (int)result.rows(); ++i)
    {
        vector<int> itemList(N);
        vector<double> rateList(N);
        for (int k = 0; k < N; ++k)
        {
            int j = result(i, k);
            itemList[k] = idItemMapList[j];
            rateList[k] = rateMat(i, j);
        }
        re[idUserMapList[i]] = make_pair(std::move(itemList), std::move(rateList));
    }*/

    //vector<T> re;
    //re.reserve(recIdxMat.size());
    //for (int i = 0; i < (int)recIdxMat.rows(); ++i)
    //{
    //    for (int k = 0; k < (int)recIdxMat.cols(); ++k)
    //    {
    //        int j = (int)recIdxMat(i, k);
    //        //re.emplace_back(idUserMapList[i], idItemMapList[j], rateMat(i, j));
    //        re.emplace_back(i, j);
    //    }
    //}

    map<int, vector<int>> re;
    for (int i = 0; i < (int)recIdxMat.rows(); ++i)
    {
        vector<int> itemList(recIdxMat.cols());
        for (int k = 0; k < (int)recIdxMat.cols(); ++k)
        {
            int j = recIdxMat(i, k);
            itemList[k] = idItemMapList[j];

        }
        re.emplace(idUserMapList[i],std::move(itemList));
    }
    return re;
}

vector<T> recommendResult(const MatrixXi& recIdxMat, const MatrixXd& rateMat, const vector<int>& idUserMap, const vector<int>& idItemMap)
{
    vector<T> re;
    re.reserve(recIdxMat.size());
    for (int i = 0; i < (int)recIdxMat.rows(); ++i)
    {
        for (int k = 0; k < (int)recIdxMat.cols(); ++k)
        {
            int j = (int)recIdxMat(i, k);
            re.emplace_back(idUserMap[i], idItemMap[j], rateMat(i, j));
        }
    }
    return re;
}

// exclude elements which is nonexistent in map
vector<T> indexTuples(const vector<T>& originData, const map<int, int>& rowMap, const map<int,int>& colMap)
{
    vector<T> indexedData;
    indexedData.reserve(originData.size());
    for (auto& t : originData)
    {
        try {
            indexedData.emplace_back(rowMap.at(t.row()), colMap.at(t.col()), t.value());
        }
        catch (const std::out_of_range& )
        {
            //std::cerr << e.what() << '\n';
            continue;
        }        
    }
    return indexedData;
}


void saveTuples(const vector<T>& data, const string& outfile)
{
    ofstream output(outfile, ios::trunc);
    for (const T& t: data)
    {
        output << t.row() << ',' << t.col() << ',' << t.value() << endl;
    }
    output.close();
    cout << "output success." << endl;
}

vector<T> choose(const vector<T>& data, const vector<int>& ids)
{
    vector<T> tmp(ids.size());
    auto itr = tmp.begin();
    for (int i: ids)
    {
        *itr = data[i];
        ++itr;
    }
    return tmp;
}


MatrixXd svdpp(const vector<T>& td, const vector<T>& vd
    , int n_rows, int n_cols, int np_ratio, int n_factor, int n_iters
    , const vector<double>& lr, const vector<double> reg)
{
    auto mmv = minmax_element(td.begin(), td.end(), [](const T& a, const T& b) { return a.value() < b.value(); });
    cout << "rate value: min " << (mmv.first)->value() << ",     max " << mmv.second->value() << endl;
    cout << "start constructing train matrix ..." << endl;

    SpMat A(n_rows, n_cols);
    A.setFromTriplets(td.begin(), td.end());

    setNegSample(A, np_ratio);
    cout << "train matrix finished." << endl;
    MatrixXd P, Q, Y;
    VectorXd bu, bi;
    cout << "start to init ..." << endl;

    initParams(&P, &Q, &bu, &bi, &Y, static_cast<int>(A.outerSize()), static_cast<int>(A.innerSize()), n_factor);
    cout << "init finished." << endl;

    double mu = A.coeffs().sum() / A.nonZeros();

    function<double(int)> learnSchedule;
    if (lr[0] == 0)
    {
        learnSchedule = [t0 = lr[1], t1 = lr[2]](int n) {return 1. / (t0 + n / t1); };
    }
    if (lr[0] == 1)
    {
        learnSchedule = [a = lr[1], b = lr[2], decay = lr[3]](int n) {return a * exp(n / decay * log(b)); };
    }
    if (lr[0] == 2)
    {
        learnSchedule = [c = lr[1]](int n) {return c; };
    }

    learningSVDpp(A, vd, &P, &Q, &bu, &bi, mu, &Y, n_iters, learnSchedule, reg[0]);
    MatrixXd rateMat = predict(P, Q, Y, constructZ1(A), bu, bi, mu);
    cout << "rateMat success!" << endl;
    return rateMat;
}




int main(int argc, char** argv)
{
    int nn = Eigen::nbThreads();
    cout << nn << endl;

    CMDLine cmdline(argc, argv);
    const string param_in_file = cmdline.registerParameter("in", "filename for training data [MANDATORY]");
    const string param_out = cmdline.registerParameter("out", "filename for out data");
    const string param_learn_rate = cmdline.registerParameter("learn_rate", "learn_rate for SGD; default=0.1");
    const string param_regular = cmdline.registerParameter("regular", "'r0,r1,r2' for SGD and ALS: r0=bias regularization, r1=1-way regularization, r2=2-way regularization");
    const string param_num_iter = cmdline.registerParameter("iter", "number of iterations; default=100");
    const string param_np_ratio = cmdline.registerParameter("np_ratio", "negative sample ratio; default=1");
    const string param_num_factor = cmdline.registerParameter("factor", "F");
    const string param_num_recommend = cmdline.registerParameter("recommend", "...");
    
    
    cout << "begin to read train data ";
    int tmp1 = 0, tmp2 = 0;
    double tmp3 = 0.0;
    char ch;
    char buffer[256];
    auto inStrs = cmdline.getStrValues(param_in_file);
    ifstream input0(inStrs[0]);
    map<int,int> userIdMap, itemIdMap;
    vector<T> tVec;
    tVec.reserve(2000000);
    while (input0.getline(buffer, 100))
    {
        stringstream ss(buffer);
        ss >> tmp1 >> ch >> tmp2 >> ch >> tmp3;
        userIdMap.insert(make_pair(tmp1,0));
        itemIdMap.insert(make_pair(tmp2,0));
        tVec.emplace_back(tmp1, tmp2, tmp3);
        if (tVec.size() % 10000 == 0) cout << '.';
    }
    input0.close(); 
    cout << "over" << endl;

    int ii = 0;
    for (auto& v : userIdMap)
    {
        v.second = ii++;
    }
    ii = 0;
    for (auto& v : itemIdMap)
    {
        v.second = ii++;
    }
    vector<int> userVec(userIdMap.size());
    vector<int> itemVec(itemIdMap.size());
    transform(userIdMap.begin(), userIdMap.end(), userVec.begin(), [](decltype(*userIdMap.begin())& a) { return a.first; });
    transform(itemIdMap.begin(), itemIdMap.end(), itemVec.begin(), [](pair<const int, int>& a) { return a.first; });

    vector<T> indexedTVec = indexTuples(tVec,userIdMap,itemIdMap);
    
    cout << "begin to read test data ";
    ifstream input1(inStrs[1]);
    vector<T> testData;
    testData.reserve(2000000);
    while (input1.getline(buffer, 100))
    {
        stringstream ss(buffer);
        ss >> tmp1 >> ch >> tmp2 >> ch >> tmp3;
        testData.emplace_back(tmp1, tmp2, tmp3);
        if (testData.size() % 10000 == 0) cout << '.';
    }
    input1.close();
    cout << "over" << endl;
    vector<T> indexedTestData = indexTuples(testData, userIdMap, itemIdMap);  
    //-----------------
    // train_test_split
    int split_K = 10; // K = 10
    vector<int> randintVec(indexedTVec.size());
    std::iota(randintVec.begin(), randintVec.end(), 0);
    srand(9);
    random_shuffle(begin(randintVec), end(randintVec));
    
    VectorXi tmp = VectorXi::LinSpaced(split_K + 1, 0, static_cast<int>(randintVec.size()) - 1);
    // the last element should be corrected.
    tmp(split_K) = (int)randintVec.size(); 

    int np_ratio = cmdline.getValue(param_np_ratio, 2);
    int n_factor = cmdline.getValue(param_num_factor, 100);
    vector<double> lr = cmdline.getDblValues(param_learn_rate);
    vector<double> reg = cmdline.getDblValues(param_regular);
    int n_Iter = cmdline.getValue(param_num_iter, 100);
    int n_recommend = cmdline.getValue(param_num_recommend, 100);
    vector<MatrixXd> rateMat(split_K);
    MatrixXd precisionMat(split_K, n_recommend);
    MatrixXd recallMat(split_K, n_recommend);
    for (int k = 0; k < split_K; ++k)
    {
        cout << "====================== " << k+1 << "-fold ======================" << endl;
        auto pickStart = randintVec.begin() + tmp(k);
        auto pickEnd = randintVec.begin() + tmp(k+1);
        vector<int> validIds{ pickStart, pickEnd };
        vector<int> trainIds(randintVec);
        auto eraseStart = trainIds.begin() + tmp(k);
        auto eraseEnd = trainIds.begin() + tmp(k+1);
        trainIds.erase(eraseStart, eraseEnd);

        vector<T> validData(choose(indexedTVec, validIds));
        vector<T> trainData(choose(indexedTVec, trainIds));

        
        rateMat[k] = svdpp(trainData, validData, (int)userVec.size(), (int)itemVec.size(),
            np_ratio, n_factor, n_Iter,lr, reg);
        auto recIdxMat = recommendIndexMatrix(rateMat[k], n_recommend);
        auto pr = precisionRecall(indexedTestData, recIdxMat);
        precisionMat.row(k) = Map<RowVectorXd>(pr.first.data(), pr.first.size());
        recallMat.row(k) = Map<RowVectorXd>(pr.second.data(), pr.second.size());
    }
    cout << "precisionMat:" << endl;
    cout << precisionMat << endl;
    cout << "recallMat:" << endl;
    cout << recallMat << endl;
    cout << "avg precision: " << precisionMat.colwise().mean() << endl;
    cout << "avg recall: " << recallMat.colwise().mean() << endl;
    /*cout << "avg precision: " << accumulate(precisionVec.begin(), precisionVec.end(), 0.0) / precisionVec.size()
        << '\t' << "avg recall: " << accumulate(recallVec.begin(), recallVec.end(), 0.0) / recallVec.size() << endl;*/


    cout << "train all data..." << endl;
    MatrixXd resultRateMat = svdpp(indexedTVec, indexedTestData, (int)userVec.size(), (int)itemVec.size(),
        np_ratio, n_factor, n_Iter, lr, reg);
    auto resultRecIdxMat = recommendIndexMatrix(resultRateMat, n_recommend);
    auto resultPR = precisionRecall(indexedTestData, resultRecIdxMat);
    Matrix2Xd resultPRMat(2, n_recommend);
    resultPRMat.row(0) = Map<RowVectorXd>(resultPR.first.data(), resultPR.first.size());
    resultPRMat.row(1) = Map<RowVectorXd>(resultPR.second.data(), resultPR.second.size());
    cout << "result precision : " << resultPRMat.row(0)<< endl;
    cout << "result recall : " << resultPRMat.row(1)<< endl;

    string outfile = cmdline.getValue(param_out, "a.out");
    saveTuples(recommendResult(resultRecIdxMat, resultRateMat, userVec, itemVec), outfile);
    return 0;
}
