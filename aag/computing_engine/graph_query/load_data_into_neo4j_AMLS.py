import pandas as pd
from neo4j import GraphDatabase
import logging
import sys
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AMLSimNeo4jLoader:
    def __init__(self, 
                 uri='bolt://219.216.65.209:7687',
                 username='neo4j', 
                 password='12345678',
                 accounts_file="/home/gaojq/AAG/aag/datasets/graphs/transaction_amlsim/1K/accounts.csv",
                 transactions_file="/home/gaojq/AAG/aag/datasets/graphs/transaction_amlsim/1K/transactions.csv"
                ):
        """
        初始化AMLSim数据加载器（Neo4j版本）
        
        Args:
            uri: Neo4j连接URI，格式为 bolt://host:port 或 neo4j://host:port
            username: 用户名
            password: 密码
            accounts_file: 账户数据文件路径
            transactions_file: 交易数据文件路径
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        
        # 数据文件路径
        self.accounts_file = accounts_file
        self.transactions_file = transactions_file
        
        self.connect_to_neo4j()
        
    def connect_to_neo4j(self):
        """连接到Neo4j"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # 验证连接
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                result.single()
            logger.info("成功连接到Neo4j")
            return True
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            if self.driver:
                self.driver.close()
                self.driver = None
            raise Exception(f"无法连接到Neo4j数据库: {e}")
    
    def clear_database(self):
        """清空数据库（可选）"""
        if not self.driver:
            logger.error("数据库连接未建立，无法清空数据库")
            return False
        try:
            with self.driver.session() as session:
                # 删除所有节点和关系
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("成功清空数据库")
                return True
        except Exception as e:
            logger.error(f"清空数据库失败: {e}")
            return False
    
    def create_constraints_and_indexes(self):
        """创建约束和索引以提升性能"""
        if not self.driver:
            logger.error("数据库连接未建立，无法创建约束和索引")
            return False
        try:
            with self.driver.session() as session:
                # 为Account节点的node_key创建唯一性约束
                try:
                    session.run("""
                        CREATE CONSTRAINT account_key_unique IF NOT EXISTS 
                        FOR (a:Account) 
                        REQUIRE a.node_key IS UNIQUE
                    """)
                    logger.info("成功创建Account.node_key唯一性约束")
                except Exception as e:
                    logger.warning(f"创建约束时出现问题（可能已存在）: {e}")
                
                # 创建node_key索引
                try:
                    session.run("CREATE INDEX account_key_index IF NOT EXISTS FOR (a:Account) ON (a.node_key)")
                    logger.info("成功创建Account.node_key索引")
                except Exception as e:
                    logger.warning(f"创建索引时出现问题（可能已存在）: {e}")
                
                # 仍然保留acct_id索引以便查询
                try:
                    session.run("CREATE INDEX account_id_index IF NOT EXISTS FOR (a:Account) ON (a.acct_id)")
                    logger.info("成功创建Account.acct_id索引")
                except Exception as e:
                    logger.warning(f"创建索引时出现问题（可能已存在）: {e}")
                
                return True
        except Exception as e:
            logger.error(f"创建约束和索引时发生错误: {e}")
            return False

    def load_accounts(self):
        """加载账户数据（节点）- 使用 'last_name first_name' 作为node_key"""
        if not self.driver:
            logger.error("数据库连接未建立，无法加载账户数据")
            return False
        try:
            # 读取账户数据
            accounts_df = pd.read_csv(self.accounts_file)
            logger.info(f"读取到 {len(accounts_df)} 条账户数据")
            
            # 替换NaN值为None
            accounts_df = accounts_df.where(pd.notna(accounts_df), None)
            
            # 创建node_key列: "last_name first_name"
            accounts_df['node_key'] = accounts_df['last_name'].astype(str) + ' ' + accounts_df['first_name'].astype(str)
            
            batch_size = 1000
            total_accounts = len(accounts_df)
            success_count = 0
            fail_count = 0
            
            with self.driver.session() as session:
                for i in range(0, total_accounts, batch_size):
                    batch = accounts_df.iloc[i:i+batch_size]
                    
                    # 将数据转换为字典列表
                    records = batch.to_dict('records')
                    
                    # 使用MERGE基于node_key
                    query = """
                    UNWIND $records AS record
                    MERGE (a:Account {node_key: record.node_key})
                    ON CREATE SET
                        a.acct_id = record.acct_id,
                        a.dsply_nm = record.dsply_nm,
                        a.type = record.type,
                        a.acct_stat = record.acct_stat,
                        a.acct_rptng_crncy = record.acct_rptng_crncy,
                        a.prior_sar_count = record.prior_sar_count,
                        a.branch_id = record.branch_id,
                        a.open_dt = record.open_dt,
                        a.close_dt = record.close_dt,
                        a.initial_deposit = record.initial_deposit,
                        a.tx_behavior_id = record.tx_behavior_id,
                        a.bank_id = record.bank_id,
                        a.first_name = record.first_name,
                        a.last_name = record.last_name,
                        a.street_addr = record.street_addr,
                        a.city = record.city,
                        a.state = record.state,
                        a.country = record.country,
                        a.zip = record.zip,
                        a.gender = record.gender,
                        a.birth_date = record.birth_date,
                        a.ssn = record.ssn,
                        a.lon = record.lon,
                        a.lat = record.lat
                    ON MATCH SET
                        a.acct_id = record.acct_id,
                        a.dsply_nm = record.dsply_nm,
                        a.type = record.type,
                        a.acct_stat = record.acct_stat,
                        a.acct_rptng_crncy = record.acct_rptng_crncy,
                        a.prior_sar_count = record.prior_sar_count,
                        a.branch_id = record.branch_id,
                        a.open_dt = record.open_dt,
                        a.close_dt = record.close_dt,
                        a.initial_deposit = record.initial_deposit,
                        a.tx_behavior_id = record.tx_behavior_id,
                        a.bank_id = record.bank_id,
                        a.first_name = record.first_name,
                        a.last_name = record.last_name,
                        a.street_addr = record.street_addr,
                        a.city = record.city,
                        a.state = record.state,
                        a.country = record.country,
                        a.zip = record.zip,
                        a.gender = record.gender,
                        a.birth_date = record.birth_date,
                        a.ssn = record.ssn,
                        a.lon = record.lon,
                        a.lat = record.lat
                    """
                    
                    try:
                        result = session.run(query, records=records)
                        summary = result.consume()
                        success_count += len(records)
                        logger.info(f"已处理 {min(i+batch_size, total_accounts)}/{total_accounts} 条账户数据")
                    except Exception as e:
                        logger.error(f"批量插入账户数据失败: {e}")
                        fail_count += len(records)
            
            logger.info(f"账户数据加载完成，成功处理 {success_count} 条，失败 {fail_count} 条")
            return True
            
        except Exception as e:
            logger.error(f"加载账户数据时发生错误: {e}")
            return False
    def load_transactions(self):
        """加载交易数据（关系）- 通过acct_id找到对应的node_key"""
        if not self.driver:
            logger.error("数据库连接未建立，无法加载交易数据")
            return False
        try:
            # 读取交易数据
            transactions_df = pd.read_csv(self.transactions_file)
            logger.info(f"读取到 {len(transactions_df)} 条交易数据")
            
            # 替换NaN值为None
            transactions_df = transactions_df.where(pd.notna(transactions_df), None)
            
            batch_size = 1000
            total_transactions = len(transactions_df)
            success_count = 0
            fail_count = 0
            
            with self.driver.session() as session:
                for i in range(0, total_transactions, batch_size):
                    batch = transactions_df.iloc[i:i+batch_size]
                    
                    # 将数据转换为字典列表
                    records = batch.to_dict('records')
                    
                    # 通过acct_id匹配节点（orig_acct和bene_acct是账户ID）
                    query = """
                    UNWIND $records AS record
                    MATCH (orig:Account {acct_id: record.orig_acct})
                    MATCH (bene:Account {acct_id: record.bene_acct})
                    CREATE (orig)-[t:TRANSACTION {
                        tran_id: record.tran_id,
                        tx_type: record.tx_type,
                        base_amt: record.base_amt,
                        tran_timestamp: record.tran_timestamp,
                        is_sar: record.is_sar,
                        alert_id: record.alert_id
                    }]->(bene)
                    """
                    
                    try:
                        result = session.run(query, records=records)
                        summary = result.consume()
                        batch_success = summary.counters.relationships_created
                        success_count += batch_success
                        logger.info(f"已处理 {min(i+batch_size, total_transactions)}/{total_transactions} 条交易数据，成功创建 {batch_success} 条关系")
                    except Exception as e:
                        logger.error(f"批量插入交易数据失败: {e}")
                        fail_count += len(records)
            
            logger.info(f"交易数据加载完成，成功插入 {success_count} 条关系，失败 {fail_count} 条")
            return True
            
        except Exception as e:
            logger.error(f"加载交易数据时发生错误: {e}")
            return False

    def verify_data(self):
        """验证数据加载结果"""
        if not self.driver:
            logger.error("数据库连接未建立，无法验证数据")
            return False
        try:
            with self.driver.session() as session:
                # 检查节点数量
                result = session.run("MATCH (a:Account) RETURN count(a) as account_count")
                account_count = result.single()["account_count"]
                logger.info(f"Account节点数量: {account_count}")
                
                # 检查关系数量
                result = session.run("MATCH ()-[t:TRANSACTION]->() RETURN count(t) as transaction_count")
                transaction_count = result.single()["transaction_count"]
                logger.info(f"TRANSACTION关系数量: {transaction_count}")
                
                # 显示几个示例节点及其node_key
                result = session.run("MATCH (a:Account) RETURN a.node_key, a.acct_id, a.first_name, a.last_name LIMIT 5")
                logger.info("示例Account节点的node_key:")
                for record in result:
                    logger.info(f"  node_key: {record['a.node_key']}, acct_id: {record['a.acct_id']}, "
                            f"name: {record['a.first_name']} {record['a.last_name']}")
                
                # 显示几个示例关系
                result = session.run("""
                    MATCH (orig:Account)-[t:TRANSACTION]->(bene:Account) 
                    RETURN orig.node_key as orig_key, bene.node_key as bene_key, t.tran_id
                    LIMIT 5
                """)
                logger.info("示例TRANSACTION关系:")
                for record in result:
                    logger.info(f"  {record['orig_key']} -> {record['bene_key']}, tran_id: {record['t.tran_id']}")
            
            return True
            
        except Exception as e:
            logger.error(f"验证数据时发生错误: {e}")
            return False

    def close_connection(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("已关闭Neo4j连接")
    
    def load_all_data(self, clear_existing=False):
        """
        加载所有数据的主流程
        
        Args:
            clear_existing: 是否清空现有数据（默认False）
        """
        try:
            # 1. 可选：清空现有数据
            if clear_existing:
                if not self.clear_database():
                    logger.warning("清空数据库失败，继续执行...")
            
            # 2. 创建约束和索引
            if not self.create_constraints_and_indexes():
                logger.warning("创建约束和索引失败，继续执行...")
            
            # 3. 加载账户数据
            if not self.load_accounts():
                return False
            
            # 4. 加载交易数据
            if not self.load_transactions():
                return False
            
            # 5. 验证数据
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
    # 注意：根据你的实际Neo4j配置修改URI
    loader = AMLSimNeo4jLoader(
        uri='bolt://202.199.13.67:7687',  # 或 'neo4j://202.199.13.174:7687'
        username='neo4j',
        password='12345678'
    )
    
    # 加载所有数据（设置clear_existing=True将清空现有数据）
    success = loader.load_all_data(clear_existing=True)
    
    if success:
        logger.info("AMLSim数据加载成功完成！")
        sys.exit(0)
    else:
        logger.error("AMLSim数据加载失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
