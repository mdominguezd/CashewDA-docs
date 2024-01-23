import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Data',
    Svg: require('@site/static/img/Data.svg').default,
    description: (
      <>
        Planet NICFI imagery from Ivory Coast (Source domain) and Tanzania (Target domain).
      </>
    ),
  },
  {
    title: 'Semantic segmentation task',
    Svg: require('@site/static/img/semanticSeg.svg').default,
    description: (
      <>
        Semantic segmentation for cashew crop mapping was performed using UNet architecture with the addition of attention gates.
      </>
    ),
  },
  {
    title: 'Domain adaptation',
    Svg: require('@site/static/img/UDA.svg').default,
    description: (
      <>
        The adoption of Unsupervised Domain Adaptation (UDA) techniques aims to align the distribution of both source and target domain.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
