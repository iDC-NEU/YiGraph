# 实现将交易图数据加载近nebulagraph
# 顶点集合：//home/chency/GraphLLM/aag/datasets/graphs/transaction_amlsim/1K/accounts.csv
# 边集合： /home/chency/GraphLLM/aag/datasets/graphs/transaction_amlsim/1K/transactions.csv
# spacename： AMLSim1K
#  顶点的tag 由/home/chency/GraphLLM/aag/datasets/graphs/transaction_amlsim/1K/accounts.csv中的字段指定，
#  顶点 id 由 acct_id 指定
#  边的 edge_type 由 /home/chency/GraphLLM/aag/datasets/graphs/transaction_amlsim/1K/transactions.csv 中的字段指定
#  边的 src_vid 是 orig_acct， dst_vid 是 bene_acct

import pandas as pd
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
import logging
import sys
import os
import time

# 配置日志
logger = logging.getLogger(__name__)

class AMLSimDataLoader:
    def __init__(self, 
                 host='localhost', 
                 port=9669, 
                 username='root', 
                 password='nebula', 
                 accounts_file="/home/chency/GraphLLM/aag/datasets/graphs/transaction_amlsim/1K/accounts.csv",
                 transactions_file = "/home/chency/GraphLLM/aag/datasets/graphs/transaction_amlsim/1K/transactions.csv"
                ):
        """
        初始化AMLSim数据加载器
        
        Args:
            host: NebulaGraph服务器地址
            port: NebulaGraph端口
            username: 用户名
            password: 密码
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.space_name = "AMLSim1K"
        self.connection_pool = None
        self.session = None
        
        # 数据文件路径
        self.accounts_file = accounts_file
        self.transactions_file = transactions_file

        self.connect_to_nebulagraph()
        
    def connect_to_nebulagraph(self):
        """连接到NebulaGraph"""
        try:
            config = Config()
            self.connection_pool = ConnectionPool()
            assert self.connection_pool.init([(self.host, self.port)], config)
            self.session = self.connection_pool.get_session(self.username, self.password)
            logger.info("成功连接到NebulaGraph")
            return True
        except Exception as e:
            logger.error(f"连接NebulaGraph失败: {e}")
            return False
    
    def create_space(self):
        """创建图空间"""
        try:
            # 检查空间是否存在，如果存在则删除
            resp = self.session.execute(f"DROP SPACE IF EXISTS {self.space_name}")
            logger.info(f"删除已存在的空间: {self.space_name}")
            
            # 创建新空间
            resp = self.session.execute(f"CREATE SPACE {self.space_name} (partition_num = 3, replica_factor = 1, vid_type = INT64)")
            if resp.is_succeeded():
                logger.info(f"成功创建空间: {self.space_name}")
                time.sleep(3)
                # 使用空间
                resp = self.session.execute(f"USE {self.space_name}")
                if resp.is_succeeded():
                    logger.info(f"成功切换到空间: {self.space_name}")
                    return True
                else:
                    logger.error(f"切换空间失败: {resp.error_msg()}")
                    return False
            else:
                logger.error(f"创建空间失败: {resp.error_msg()}")
                return False
        except Exception as e:
            logger.error(f"创建空间时发生错误: {e}")
            return False
    
    def use_space(self):
        resp = self.session.execute(f"USE {self.space_name}")
        if resp.is_succeeded():
            logger.info(f"成功切换到空间: {self.space_name}")
            return True
        else:
            logger.error(f"切换空间失败: {resp.error_msg()}")
            return False

    
    def create_schema(self):
        self.use_space()
        """创建图模式（Tag和Edge Type）"""
        try:
            # 创建Account Tag
            account_tag_schema = """
            CREATE TAG account(
                acct_id INT,
                dsply_nm STRING,
                type STRING,
                acct_stat STRING,
                acct_rptng_crncy STRING,
                prior_sar_count BOOL,
                branch_id INT,
                open_dt INT,
                close_dt INT,
                initial_deposit DOUBLE,
                tx_behavior_id STRING,
                bank_id STRING,
                first_name STRING,
                last_name STRING,
                street_addr STRING,
                city STRING,
                state STRING,
                country STRING,
                zip STRING,
                gender STRING,
                birth_date STRING,
                ssn STRING,
                lon DOUBLE,
                lat DOUBLE
            )
            """
            resp = self.session.execute(account_tag_schema)
            if resp.is_succeeded():
                logger.info("成功创建Account Tag")
            else:
                logger.error(f"创建Account Tag失败: {resp.error_msg()}")
                return False
            
            # 创建Transaction Edge Type
            transaction_edge_schema = """
            CREATE EDGE transaction(
                tran_id INT,
                tx_type STRING,
                base_amt DOUBLE,
                tran_timestamp STRING,
                is_sar BOOL,
                alert_id INT
            )
            """
            resp = self.session.execute(transaction_edge_schema)
            if resp.is_succeeded():
                logger.info("成功创建Transaction Edge Type")
                # 等待模式生效
                time.sleep(30)   
                return True
            else:
                logger.error(f"创建Transaction Edge Type失败: {resp.error_msg()}")
                return False
 
        except Exception as e:
            logger.error(f"创建模式时发生错误: {e}")
            return False
    
    def load_accounts(self):
        self.use_space()
        resp = self.session.execute("SHOW TAGS")
        fail_insert = 0
        for row in resp.rows():
            print("当前图空间已定义的tag:", row.values[0].get_sVal().decode("utf-8"))
        """加载账户数据（顶点）"""
        try:
            # 读取账户数据
            accounts_df = pd.read_csv(self.accounts_file)
            logger.info(f"读取到 {len(accounts_df)} 条账户数据")
            
            # 批量插入顶点
            batch_size = 1000
            total_accounts = len(accounts_df)
            
            for i in range(0, total_accounts, batch_size):
                batch = accounts_df.iloc[i:i+batch_size]
                
                # 构建插入语句
                insert_statements = []
                for _, row in batch.iterrows():
                    # 处理空值
                    values = []
                    for col in accounts_df.columns:
                        if pd.isna(row[col]):
                            values.append('""')
                        elif isinstance(row[col], str) or col == "zip": 
                            values.append(f'"{row[col]}"')
                        elif isinstance(row[col], bool):
                            values.append(str(row[col]).lower())
                        else:
                            values.append(str(row[col]))
                    
                    insert_stmt = f'INSERT VERTEX account({", ".join(accounts_df.columns)}) VALUES {int(row["acct_id"])}:({", ".join(values)})'
                    insert_statements.append(insert_stmt)
                
                # 执行批量插入
                for stmt in insert_statements:
                    resp = self.session.execute(stmt)
                    print(stmt)
                    if not resp.is_succeeded():
                        logger.warning(f"插入顶点失败: {resp.error_msg()}")
                        fail_insert += 1

                logger.info(f"已处理 {min(i+batch_size, total_accounts)}/{total_accounts} 条账户数据")
            
            success_insert = total_accounts - fail_insert
            logger.info(f"账户数据加载完成， 成功插入{success_insert}条账户数据， 失败插入{fail_insert}条账户数据")
            return True
            
        except Exception as e:
            logger.error(f"加载账户数据时发生错误: {e}")
            return False
    
    def load_transactions(self):
        self.use_space()
        fail_insert = 0
        """加载交易数据（边）"""
        try:
            # 读取交易数据
            transactions_df = pd.read_csv(self.transactions_file)
            logger.info(f"读取到 {len(transactions_df)} 条交易数据")
            
            # 批量插入边
            batch_size = 1000
            total_transactions = len(transactions_df)
            
            for i in range(0, total_transactions, batch_size):
                batch = transactions_df.iloc[i:i+batch_size]
                
                # 构建插入语句
                insert_statements = []
                for _, row in batch.iterrows():
                    # 处理空值，排除orig_acct和bene_acct字段
                    edge_properties = [col for col in transactions_df.columns if col not in ['orig_acct', 'bene_acct']]
                    values = []
                    for col in edge_properties:
                        if pd.isna(row[col]):
                            values.append('""')
                        elif isinstance(row[col], str):
                            values.append(f'"{row[col]}"')
                        elif isinstance(row[col], bool):
                            values.append(str(row[col]).lower())
                        else:
                            values.append(str(row[col]))
                            
                    rank = int(row["tran_id"])
                    insert_stmt = f'INSERT EDGE transaction({", ".join(edge_properties)}) VALUES {int(row["orig_acct"])} -> {int(row["bene_acct"])}@{rank}:({", ".join(values)})'
                    insert_statements.append(insert_stmt)
                
                # 执行批量插入
                for stmt in insert_statements:
                    resp = self.session.execute(stmt)
                    if not resp.is_succeeded():
                        logger.warning(f"插入边失败: {resp.error_msg()}")
                
                logger.info(f"已处理 {min(i+batch_size, total_transactions)}/{total_transactions} 条交易数据")
            
            logger.info("交易数据加载完成")
            return True
            
        except Exception as e:
            logger.error(f"加载交易数据时发生错误: {e}")
            return False
    
    def verify_data(self):
        """验证数据加载结果"""
        try:
            # 检查顶点数量
            resp = self.session.execute("MATCH (n:account) RETURN count(n) as account_count")
            if resp.is_succeeded():
                account_count = resp.row_values(0)[0].as_int()
                logger.info(f"account顶点数量: {account_count}")
            
            # 检查边数量
            resp = self.session.execute("MATCH ()-[e:transaction]->() RETURN count(e) as transaction_count")
            if resp.is_succeeded():
                transaction_count = resp.row_values(0)[0].as_int()
                logger.info(f"transaction边数量: {transaction_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"验证数据时发生错误: {e}")
            return False
    
    def wait_for_tag_ready(self, tag_name, timeout=30):
        for _ in range(timeout):
            resp = self.session.execute(f"DESCRIBE TAG {tag_name}")
            if resp.is_succeeded():
                return True
            time.sleep(10)
        return False
    
    def wait_for_edge_ready(self, edge_name, timeout=10):
        for _ in range(timeout):
            resp = self.session.execute(f"DESCRIBE EDGE {edge_name}")
            if resp.is_succeeded():
                return True
            time.sleep(1)
        return False


    def close_connection(self):
        """关闭连接"""
        if self.session:
            self.session.release()
        if self.connection_pool:
            self.connection_pool.close()
        logger.info("已关闭NebulaGraph连接")
    
    def load_all_data(self):
        """加载所有数据的主流程"""
        try:
            
            # 2. 创建空间
            if not self.create_space():
                return False
            
            # 3. 创建模式
            if not self.create_schema():
                return False
            
            # 4. 加载账户数据
            if not self.load_accounts():
                return False
            
            # 5. 加载交易数据
            if not self.load_transactions():
                return False
            
            # 6. 验证数据
            if not self.verify_data():
                return False
            
            logger.info("所有数据加载完成！")
            return True
            
        except Exception as e:
            logger.error(f"加载数据时发生错误: {e}")
            return False
        finally:
            self.close_connection()

def main():
    """主函数"""
    # 创建数据加载器
    loader = AMLSimDataLoader()
    
    # 加载所有数据
    success = loader.load_all_data()
    
    if success:
        logger.info("AMLSim数据加载成功完成！")
        sys.exit(0)
    else:
        logger.error("AMLSim数据加载失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()

